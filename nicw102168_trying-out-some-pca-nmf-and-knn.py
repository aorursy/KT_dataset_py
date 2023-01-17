import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline   

plt.rcParams['image.cmap'] = 'gray'



def read_vectors(*filenames):

    data = np.vstack(

        tuple(np.fromfile(filename, dtype=np.uint8).reshape(-1,401)

                      for filename in filenames))

    return data[:,1:], data[:,0]



X_train, y_train = read_vectors(*[

    "../input/snake-eyes/snakeeyes_{:02d}.dat".format(nn) for nn in range(2)])

X_test, y_test = read_vectors("../input/snake-eyes/snakeeyes_test.dat")
from sklearn.decomposition import PCA

pca = PCA(random_state=0, whiten=True)

pca.fit(X_train);
exp_var_cum = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(exp_var_cum.size), exp_var_cum)

plt.grid()
plt.plot(range(exp_var_cum.size), exp_var_cum, '-+')

plt.grid()

plt.xlim(15,105)

plt.ylim(0.6,0.95);
plt.figure(figsize=(20,10))

for k in range(40):

    plt.subplot(4,10,k+1)

    plt.imshow(pca.components_[k].reshape(20,20))

    plt.axis('off')
pca = PCA(n_components=90, random_state=0, whiten=True)

pca.fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
X_reconstructed_pca = pca.inverse_transform(X_test_pca)



plt.figure(figsize=(20,10))

for k in range(20):

    plt.subplot(4, 10, k*2 + 1)

    plt.imshow(X_test[k].reshape(20,20))

    plt.axis('off')

    plt.subplot(4, 10, k*2 + 2)

    plt.imshow(X_reconstructed_pca[k].reshape(20,20))

    plt.axis('off')
KNN_PCA_TRAIN_SIZE = 200000

KNN_PCA_TEST_SIZE = 200



from sklearn.neighbors import KNeighborsClassifier

temp = []

for i in [1, 5]:

    knn_pca = KNeighborsClassifier(n_neighbors=i, n_jobs=8)

    knn_pca.fit(X_train_pca[:KNN_PCA_TRAIN_SIZE], y_train[:KNN_PCA_TRAIN_SIZE])

    train_score_pca = knn_pca.score(X_train_pca[:KNN_PCA_TEST_SIZE], y_train[:KNN_PCA_TEST_SIZE])

    test_score_pca = knn_pca.score(X_test_pca[:KNN_PCA_TEST_SIZE], y_test[:KNN_PCA_TEST_SIZE])

    li = [i,train_score_pca,test_score_pca]

    temp.append(li)

temp
NMF_TRAIN_SIZE = 100000



from sklearn.decomposition import NMF

nmf = NMF(n_components=90, random_state=0)

nmf.fit(X_train[:NMF_TRAIN_SIZE])
plt.figure(figsize=(20,10))

for k in range(40):

    plt.subplot(4, 10, k + 1)

    plt.imshow(nmf.components_[k].reshape(20,20))

    plt.axis('off')
X_train_nmf = nmf.transform(X_train)

X_test_nmf = nmf.transform(X_test)

X_reconstructed = nmf.inverse_transform(X_test_nmf)
plt.figure(figsize=(20,10))

for k in range(20):

    plt.subplot(4, 10, k*2 + 1)

    plt.imshow(X_test[k].reshape(20,20))

    plt.axis('off')

    plt.subplot(4, 10, k*2 + 2)

    plt.imshow(X_reconstructed[k].reshape(20,20))

    plt.axis('off')
KNN_NMF_TRAIN_SIZE = 200000

KNN_NMF_TEST_SIZE = 200



from sklearn.neighbors import KNeighborsClassifier

temp = []

for i in [1, 5]:

    knn_nmf = KNeighborsClassifier(n_neighbors=i, n_jobs=8)

    knn_nmf.fit(X_train_nmf[:KNN_NMF_TRAIN_SIZE], y_train[:KNN_NMF_TRAIN_SIZE])

    train_score_nmf = knn_nmf.score(X_train_nmf[:KNN_NMF_TEST_SIZE], y_train[:KNN_NMF_TEST_SIZE])

    test_score_nmf = knn_nmf.score(X_test_nmf[:KNN_NMF_TEST_SIZE], y_test[:KNN_NMF_TEST_SIZE])

    li = [i,train_score_nmf,test_score_nmf]

    temp.append(li)

temp