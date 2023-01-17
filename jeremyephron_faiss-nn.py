!pip3 install faiss-gpu
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise
def create_data(n_train=1600, n_dev=100, n_test=250, d=768, n_label_fns=10, abstain_prob=0.8):
    np.random.seed(137)  # make reproducible
    train_embeddings = np.random.normal(scale=0.1, size=(n_train, d)).astype('float32')
    L_train = np.random.choice([0, -1, 1], size=(n_train, n_label_fns), replace=True, 
                               p=[abstain_prob, (1 - abstain_prob) / 2, (1 - abstain_prob) / 2])

    np.random.seed(138)
    dev_embeddings = np.random.normal(scale=0.1, size=(n_dev, d)).astype('float32')
    L_dev = np.random.choice([0, -1, 1], size=(n_dev, n_label_fns), replace=True, 
                             p=[abstain_prob, (1 - abstain_prob) / 2, (1 - abstain_prob) / 2])

    np.random.seed(139)
    test_embeddings = np.random.normal(scale=0.1, size=(n_test, d)).astype('float32')
    L_test = np.random.choice([0, -1, 1], size=(n_test, n_label_fns), replace=True, 
                              p=[abstain_prob, (1 - abstain_prob) / 2, (1 - abstain_prob) / 2])
    
    return train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test
def faiss_init(train_embeddings, L_train, gpu=False):
    """One time setup."""
    
    # Create one index for each labeling function
    d = train_embeddings.shape[1]
    m = L_train.shape[1]
    label_fn_indexes = [faiss.IndexFlatL2(d) for i in range(m)]
    if gpu:
        res = faiss.StandardGpuResources()  # use a single GPU
        label_fn_indexes = [faiss.index_cpu_to_gpu(res, 0, x) for x in label_fn_indexes]
    
    # Add the embeddings to the index for which the labeling function is supported
    for i in range(m):
        support = np.argwhere(L_train[:, i] != 0).flatten()
        label_fn_indexes[i].add(train_embeddings[support])
        
    return label_fn_indexes


def faiss_nn_query(indexes, embs_mat, L_mat):
    """Helper function to perform nearest-neighbor queries reusing indexes."""
    
    m = L_mat.shape[1]
    
    mat_abstains = [
        np.argwhere(L_mat[:, i] == 0).flatten()
        for i in range(m)
    ]
    res = [indexes[i].search(embs_mat[mat_abstains[i]], 1) for i in range(m)]
    return res, sum(len(mat_abstains[i]) for i in range(m))
def old_nn_query(L_train, embs_train, L_mat, embs_mat):
    mat_to_train_sims = pairwise.euclidean_distances(embs_mat, embs_train)
    
    m = L_mat.shape[1]
    expanded_L_mat = np.copy(L_mat)

    train_support_pos = [
        np.argwhere(L_train[:, i] == 1).flatten()
        for i in range(m)
    ]
    train_support_neg = [
        np.argwhere(L_train[:, i] == -1).flatten()
        for i in range(m)
    ]

    mat_abstains = [
        np.argwhere(L_mat[:, i] == 0).flatten()
        for i in range(m)
    ]

    pos_dists = [
        mat_to_train_sims[mat_abstains[i]][:, train_support_pos[i]]
        for i in range(m)
    ]
    neg_dists = [
        mat_to_train_sims[mat_abstains[i]][:, train_support_neg[i]]
        for i in range(m)
    ]

    closest_pos = [
        np.max(pos_dists[i], axis=1)
        if pos_dists[i].shape[1] > 0 else np.full(mat_abstains[i].shape, -1)
        for i in range(m)
    ]
    closest_neg = [
        np.max(neg_dists[i], axis=1)
        if neg_dists[i].shape[1] > 0 else np.full(mat_abstains[i].shape, -1)
        for i in range(m)
    ]

    return mat_abstains, closest_pos, closest_neg
sizes = list(range(1600, 25600, 800)) + list(range(25600, 102400, 1600)) + list(range(102400, 204801, 3200))
times_old = []
times_faiss = []
times_faiss_gpu = []
for sz in sizes:
    train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=sz,
                                                                                            n_dev=int(0.05*sz),
                                                                                            n_test=int(0.10*sz))
    print('size:', sz)
    
    if sz <= 25600:
        start = time.time()
        old_nn_query(L_train, train_embeddings, L_train, train_embeddings)
        old_nn_query(L_train, train_embeddings, L_dev, dev_embeddings)
        old_nn_query(L_train, train_embeddings, L_test, test_embeddings)
        end = time.time()
        times_old.append(end - start)
        print('old:', times_old[-1])
    
    if sz <= 49600:
        start = time.time()
        indexes = faiss_init(train_embeddings, L_train)
        faiss_nn_query(indexes, train_embeddings, L_train)
        faiss_nn_query(indexes, dev_embeddings, L_dev)
        faiss_nn_query(indexes, test_embeddings, L_test)
        end = time.time()
        times_faiss.append(end - start)
        print('cpu:', times_faiss[-1])
    
    start = time.time()
    indexes = faiss_init(train_embeddings, L_train, gpu=True)
    faiss_nn_query(indexes, train_embeddings, L_train)
    faiss_nn_query(indexes, dev_embeddings, L_dev)
    faiss_nn_query(indexes, test_embeddings, L_test)
    end = time.time()
    times_faiss_gpu.append(end - start)
    print('gpu:', times_faiss_gpu[-1])
# plt.plot(sizes[:len(times_old)], times_old, 'o')
# plt.plot(sizes[:len(times_old)], times_faiss[:len(times_old)], 'o')
# # plt.plot(sizes[:len(times_faiss_gpu)], times_faiss_gpu, 'o')
# plt.legend(['old', 'FAISS cpu'])
# plt.ylabel('seconds')
# plt.xlabel('size of training set')
# plt.show()

a2, a1, a0 = np.polyfit(sizes[:len(times_old)], times_old, 2)
b2, b1, b0 = np.polyfit(sizes[:len(times_faiss)], times_faiss, 2)
c2, c1, c0 = np.polyfit(sizes[:len(times_faiss_gpu)], times_faiss_gpu, 2)
plt.plot(sizes, a2 * (np.array(sizes)**2) + a1 * np.array(sizes) + a0)
plt.plot(sizes, b2 * (np.array(sizes)**2) + b1 * np.array(sizes) + b0)
plt.plot(sizes, c2 * (np.array(sizes)**2) + c1 * np.array(sizes) + c0)
plt.title('Projections (actual times for GPU)')
plt.legend(['old', 'FAISS cpu', 'FAISS gpu'])
plt.ylabel('seconds')
plt.xlabel('size of training set')
plt.show()
n_label_fns = range(1, 258, 8)
times = []
for n in n_label_fns:
    train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_label_fns=n)
    start = time.time()
    indexes = faiss_init(train_embeddings, L_train)
    _, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
    end = time.time()
    times.append(end - start)
    print(end-start)
    print(n_queries)
    
gpu_times = []
for n in n_label_fns:
    train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_label_fns=n)
    start = time.time()
    indexes = faiss_init(train_embeddings, L_train, gpu=True)
    _, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
    end = time.time()
    gpu_times.append(end - start)
    print(end-start)
    print(n_queries)
m1, b1 = np.polyfit(n_label_fns, times, 1)
plt.plot(n_label_fns, times, 'o')

m2, b2 = np.polyfit(n_label_fns, gpu_times, 1)
plt.plot(n_label_fns, gpu_times, 'o')

plt.plot(n_label_fns, m1 * np.array(n_label_fns) + b1)
plt.plot(n_label_fns, m2 * np.array(n_label_fns) + b2)
plt.legend(['cpu', 'gpu', f'y={m1:.3}x+{b1:.3}', f'y={m2:.3}x+{b2:.3}'])
plt.xlabel('number of labeling functions')
plt.ylabel('seconds')
plt.show()
sizes = range(1600, 100001, 800)
times = []
for sz in sizes:
    train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=sz)
    start = time.time()
    indexes = faiss_init(train_embeddings, L_train)
    _, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
    end = time.time()
    times.append(end - start)
    print(end-start)
    print(n_queries)
    
gpu_times = []
for sz in sizes:
    train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=sz)
    start = time.time()
    indexes = faiss_init(train_embeddings, L_train, gpu=True)
    _, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
    end = time.time()
    gpu_times.append(end - start)
    print(end-start)
    print(n_queries)
m1, b1 = np.polyfit(sizes, times, 1)
plt.plot(sizes, times, 'o')

m2, b2 = np.polyfit(sizes, gpu_times, 1)
plt.plot(sizes, gpu_times, 'o')

plt.plot(sizes, m1 * np.array(sizes) + b1)
plt.plot(sizes, m2 * np.array(sizes) + b2)
plt.legend(['cpu', 'gpu', f'y={m1:.3}x+{b1:.3}', f'y={m2:.3}x+{b2:.3}'])
plt.xlabel('size of training set')
plt.ylabel('seconds')
plt.show()
dims = range(600, 3200, 100)
times = []
for d in dims:
    train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=10000, d=d)
    start = time.time()
    indexes = faiss_init(train_embeddings, L_train)
    _, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
    end = time.time()
    times.append(end - start)
    print(end-start)
    print(n_queries)
    
gpu_times = []
for d in dims:
    train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=10000, d=d)
    start = time.time()
    indexes = faiss_init(train_embeddings, L_train)
    _, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
    end = time.time()
    gpu_times.append(end - start)
    print(end-start)
    print(n_queries)
m1, b1 = np.polyfit(dims, times, 1)
plt.plot(dims, times, 'o')

m2, b2 = np.polyfit(dims, gpu_times, 1)
plt.plot(dims, gpu_times, 'o')

plt.plot(dims, m1 * np.array(dims) + b1)
plt.plot(dims, m2 * np.array(dims) + b2)
plt.legend(['cpu', 'gpu', f'y={m1:.3}x+{b1:.3}', f'y={m2:.3}x+{b2:.3}'])
plt.xlabel('dimension of embeddings')
plt.ylabel('seconds')
plt.show()
sizes = range(1600, 12800, 800)
times = []
queries = []
for sz in sizes:
    train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=12800, n_test=sz)
    start = time.time()
    indexes = faiss_init(train_embeddings, L_train)
    _, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
    end = time.time()
    times.append(end - start)
    queries.append(n_queries)
    print(end-start)
    print(n_queries)
    
gpu_times = []
gpu_queries = []
for sz in sizes:
    train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=12800, n_test=sz)
    start = time.time()
    indexes = faiss_init(train_embeddings, L_train, gpu=True)
    _, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
    end = time.time()
    gpu_times.append(end - start)
    gpu_queries.append(n_queries)
    print(end-start)
    print(n_queries)
m1, b1 = np.polyfit(queries, times, 1)
plt.plot(queries, times, 'o')

m2, b2 = np.polyfit(queries, gpu_times, 1)
plt.plot(queries, gpu_times, 'o')

plt.plot(queries, m1 * np.array(queries) + b1)
plt.plot(queries, m2 * np.array(queries) + b2)
plt.legend(['cpu', 'gpu', f'y={m1:.3}x+{b1:.3}', f'y={m2:.3}x+{b2:.3}'])
plt.xlabel('number of queries')
plt.ylabel('seconds')
plt.show()
train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=1586, n_label_fns=10, n_test=1922, abstain_prob=0.837)
start = time.time()
indexes = faiss_init(train_embeddings, L_train)
_, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
end = time.time()
print('cpu:', end-start)

start = time.time()
indexes = faiss_init(train_embeddings, L_train, gpu=True)
_, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
end = time.time()
print('gpu:', end-start)
# print(n_queries)
train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=17970, n_label_fns=4, n_test=20245, abstain_prob=0.507)
start = time.time()
indexes = faiss_init(train_embeddings, L_train)
_, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
end = time.time()
print('cpu:', end-start)

start = time.time()
indexes = faiss_init(train_embeddings, L_train, gpu=True)
_, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
end = time.time()
print('gpu:', end-start)
# print(n_queries)
train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=187, n_label_fns=103, n_test=290, abstain_prob=0.912)
start = time.time()
indexes = faiss_init(train_embeddings, L_train)
_, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
end = time.time()
print('cpu:', end-start)

start = time.time()
indexes = faiss_init(train_embeddings, L_train, gpu=True)
_, n_queries = faiss_nn_query(indexes, test_embeddings, L_test)
end = time.time()
print('gpu:', end-start)
# print(n_queries)
def faiss_init(train_embeddings, L_train, metric='L2', gpu=False):
    """One time setup."""
    
    if metric == 'cosine':
        # Copy because faiss.normalize_L2() modifies the original
        train_embeddings = np.copy(train_embeddings)
        
        # Normalize the vectors before adding to the index
        faiss.normalize_L2(train_embeddings)
    
    d = train_embeddings.shape[1]
    m = L_train.shape[1]
    
    if metric == 'cosine':
        label_fn_indexes = [faiss.IndexFlatIP(d) for i in range(m)]  # use IndexFlatIP (inner product)
    else:  # 'L2':
        label_fn_indexes = [faiss.IndexFlatL2(d) for i in range(m)]
        
    if gpu:
        res = faiss.StandardGpuResources()
        label_fn_indexes = [faiss.index_cpu_to_gpu(res, 0, x) for x in label_fn_indexes]
    
    for i in range(m):
        support = np.argwhere(L_train[:, i] != 0).flatten()
        label_fn_indexes[i].add(train_embeddings[support])
        
    return label_fn_indexes


def faiss_nn_query(indexes, embs_mat, L_mat, metric='L2'):
    """Helper function to perform nearest-neighbor queries reusing indexes."""
    
    m = L_mat.shape[1]
    
    mat_abstains = [
        np.argwhere(L_mat[:, i] == 0).flatten()
        for i in range(m)
    ]
    
    res = []
    for i in range(m):
        if metric == 'cosine':
            embs_query = np.copy(embs_mat[mat_abstains[i]])
            faiss.normalize_L2(embs_query)
        else: # 'L2'
            embs_query = embs_mat[mat_abstains[i]]
        res.append(indexes[i].search(embs_query, 1))
        
    return res, sum(len(mat_abstains[i]) for i in range(m))
train_embeddings, L_train, dev_embeddings, L_dev, test_embeddings, L_test = create_data(n_train=10000)
indexes = faiss_init(train_embeddings, L_train, metric='cosine')
res, n_queries = faiss_nn_query(indexes, test_embeddings, L_test, metric='cosine')
# res is a list of length 2, where the first element is an array of distances,
# and the second element is a list of indices of the nearest neighbors