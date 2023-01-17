# In this kernel, an algorithm of multi-instance learning is designed

# Import modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time



# Sigmoid function

from scipy.stats import logistic

from scipy.sparse import *



# Cosine similarity

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import normalize

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix



# Modules for parallelizing the code

from joblib import Parallel, delayed

import multiprocessing

from tqdm.notebook import tqdm



# Module for clearing the memory

import gc



# Module for GPU computing

import torch

import torch.nn as nn
def rbf_similarity_loss(W, batch_start, batch_end):

    # Dimensions

    V = W.shape[0]

    d = W.shape[1]

    batchSize = batch_end - batch_start

    

    F1 = (W**2).sum(dim = 1).view(V,1) @ torch.ones((1,batchSize)).cuda()

    F2 = (W[batch_start:batch_end,:]**2).sum(dim = 1).view(batchSize,1) @ torch.ones((1,V)).cuda()

    F3 = W @ W[batch_start:batch_end,:].T

    return(torch.exp(-(F1 + F2.T - 2 * F3)))
def rbf_similarity(W, W0):

    # Dimensions

    V = W.shape[0]

    d = W.shape[1]

    batchSize = W0.shape[0]

    

    F1 = (W**2).sum(dim = 1).view(V,1) @ torch.ones((1,batchSize)).cuda()

    F2 = (W0**2).sum(dim = 1).view(batchSize,1) @ torch.ones((1,V)).cuda()

    F3 = W @ W0.T

    return(torch.exp(-(F1 + F2.T - 2 * F3)))
def rbf_similarity_cpu(W, batch_start, batch_end):

    # Dimensions

    V = W.shape[0]

    d = W.shape[1]

    batchSize = batch_end - batch_start



    F1 = (W**2).sum(axis = 1).reshape(V,1) @ np.ones((1,batchSize))

    F2 = (W[batch_start:batch_end,:]**2).sum(axis = 1).reshape(batchSize,1) @ np.ones((1,V))

    F3 = W @ W[batch_start:batch_end,:].T

    return(np.exp(-(F1 + F2.T - 2 * F3)))
def first_term_torch(X, sig, numChunk = 5, type_sim = 'cosine', batchSize = 2**9):

    # Dimensions

    N = X.shape[0]

    d = X.shape[1]

    

    # Reshape sigmoid

    sig = sig.view(N,1)

    grad_sig = sig * (1 - sig)

    

    if type_sim == 'cosine':

        # Norm

        inv_norm = torch.norm(X, p = 2, dim = 1).pow_(-1)

        

        # Split the data

        knots = list(torch.arange(0, N, step = -(-N//numChunk)).cuda())

        knots.append(N)

        chunks = []

        for i in range(1,len(knots)):

            chunks.append([knots[i-1],knots[i]])

            

        # First term

        A1 = torch.zeros((1,d)).cuda()

        A2 = torch.zeros((d,1)).cuda()

        A3 = torch.zeros((1,d)).cuda()

        for chunk in chunks:

            chunkSize = (chunk[1]-chunk[0]).int()

            ii = torch.mm(torch.ones((2,1)),torch.arange(0,chunkSize).view(1,chunkSize).float()).long().cuda()

            diag_inv_norm = torch.sparse.FloatTensor(ii, inv_norm[chunk[0]:chunk[1]].flatten().float(), torch.Size([chunkSize,chunkSize])) 

            Xnorm = torch.sparse.mm(diag_inv_norm, X[chunk[0]:chunk[1],:])

            

            A1 += torch.t(sig[chunk[0]:chunk[1]]**2)@Xnorm

            A2 += torch.sum(Xnorm, axis = 0).view(d,1)

            A3 += torch.t(sig[chunk[0]:chunk[1]])@Xnorm

            

        first_term = A1@A2 + torch.t(A1@A2) - 2*A3@torch.t(A3)

    elif type_sim == 'rbf':

        knots = list(torch.arange(0, N, step = batchSize).cuda())

        knots.append(N)

        batches = []

        for i in range(1,len(knots)):

            batches.append([knots[i-1],knots[i]])

        

        A = torch.zeros((N,1)).cuda().float()

        B = torch.zeros((N,1)).cuda().float()

        for batch in batches:

            D = rbf_similarity_loss(X, batch[0], batch[1])

            A += D @ (torch.ones(((batch[1]-batch[0]),1)).cuda().float())

            B += D @ sig[batch[0]: batch[1]]

        first_term = 2* sig.T @ ((sig * A) - B)

    return(1/(N**2) * first_term)
def second_term_torch(X, y, z, H, sig, theta):

    N = X.shape[0]

    d = X.shape[1]

    K = H.shape[1]

    

    # Reshape

    y = y.view(N,1)

    theta = theta.view(d,1)



    label = torch.sign(torch.sparse.mm(H.t(), y))

    

    return(torch.mean((H.t()@(z*sig) - label)**2))
# Testing the loss function

def first_test(X, y, H, theta, type_sim = 'rbf'):

    N = X.shape[0]

    d = X.shape[1]

    K = H.shape[1]



    theta = theta.reshape(d,1)

    if type_sim == 'rbf':

        similarities = rbf_similarity_cpu(X, 0, N)

    elif type_sim == 'cosine':

        similarities = cosine_similarity(X)

    sig = logistic.cdf(X @ theta)



    first_term = 0

    for i in range(N):

        for j in range(N):

            first_term += similarities[i,j] * (sig[i] - sig[j])**2

    first_term = first_term/(N**2)

    return(first_term)



def second_test(X, y, H, theta):

    N = X.shape[0]

    d = X.shape[1]

    K = H.shape[1]



    theta = theta.reshape(d,1)



    second_term = 0

    sig = logistic.cdf(X @ theta)



    true_label = np.sign(H.T @ y)

    idx = [0] + list(np.cumsum(np.sum(H.toarray(), axis = 0)))

    blocks = [[int(idx[i]), int(idx[i+1])] for i in range(len(idx)-1)]

    for i, block in enumerate(blocks):

        second_term += (np.mean(sig[block[0]:block[1]]) - true_label[i])**2

    

    # aver_sig_doc = np.array(H.T @ sig) / np.sum(H, axis = 0).reshape(K,1)

    # second_term = np.mean((aver_sig_doc - true_label)**2)

    second_term = 1/K * second_term

    return(second_term)



# Test data

N = 20

d = 5

lamb = 1

np.random.seed(1)

X = np.random.random((N, d))

y = np.array(list([np.ones(5)] + [np.zeros(5)]) * 2).flatten()

H = block_diag([np.ones((5,1))] * int(N/5))

K = H.shape[1]



np.random.seed(2)

theta = np.random.normal(0, 1, (1,d))



print('Test functions')

print(first_test(X, y, H, theta, type_sim = 'rbf'))

print(second_test(X, y, H, theta))



X_gpu = torch.from_numpy(X).cuda().float()

y_gpu = torch.from_numpy(y).cuda().float()

theta_gpu = torch.from_numpy(theta).cuda().float()



values = H.data

indices = np.vstack((H.row, H.col))



i = torch.LongTensor(indices)

v = torch.FloatTensor(values)

shape = H.shape

        

H_gpu = torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()

z_gpu = torch.sparse.mm(H_gpu, 1/torch.sparse.sum(H_gpu, dim = 0).to_dense().view(K,1)).view(N,1)



sig = 1/(1 + torch.exp(-X_gpu @ theta_gpu.view(d,1))).view(N,1)



print('Used functions')

print(first_term_torch(X_gpu, sig, numChunk = 5, type_sim = 'rbf'))

print(second_term_torch(X_gpu, y_gpu, z_gpu, H_gpu, sig, theta_gpu))
def first_term_gradient_torch(X, sig, numChunk = 5, type_sim = 'cosine', batchSize = 2**9):

    # Dimensions

    N = X.shape[0]

    d = X.shape[1]

    batchSize = min(N, batchSize)

    

    # Reshape sigmoid

    sig = sig.view(N,1)

    grad_sig = sig * (1 - sig)

    

    if type_sim == 'cosine':

        # Norm

        inv_norm = torch.norm(X, p = 2, dim = 1).pow_(-1)

        

        # Split the data

        knots = list(torch.arange(0, N, step = -(-N//numChunk)).cuda())

        knots.append(N)

        chunks = []

        for i in range(1,len(knots)):

            chunks.append([knots[i-1],knots[i]])

        

        # First term

        A1 = torch.zeros((d, d)).cuda()

        A3 = torch.zeros((d, d)).cuda()

        A2 = torch.zeros((d, 1)).cuda()

        A4 = torch.zeros((d, 1)).cuda()

        for chunk in chunks:

            chunkSize = (chunk[1]-chunk[0]).int()

            ii = torch.mm(torch.ones((2,1)),torch.arange(0,chunkSize).view(1,chunkSize).float()).long().cuda()

            diag_sig = torch.sparse.FloatTensor(ii, sig[chunk[0]:chunk[1]].flatten().float(), torch.Size([chunkSize,chunkSize])) 

            diag_grad_sig = torch.sparse.FloatTensor(ii, (sig[chunk[0]:chunk[1]]*(1-sig[chunk[0]:chunk[1]])).flatten().float(), torch.Size([chunkSize,chunkSize]))

            diag_inv_norm = torch.sparse.FloatTensor(ii, inv_norm[chunk[0]:chunk[1]].flatten().float(), torch.Size([chunkSize,chunkSize])) 

            Xnorm = torch.sparse.mm(diag_inv_norm, X[chunk[0]:chunk[1],:])

            Xbar = torch.sparse.mm(diag_grad_sig, X[chunk[0]:chunk[1],:])

            

            A = torch.sparse.mm(diag_sig, Xnorm)

            A1 += torch.t(Xbar)@A

            A2 += torch.t(Xnorm)@torch.ones((chunkSize,1)).cuda()

            A3 += torch.t(Xbar)@Xnorm

            A4 += torch.t(Xnorm)@sig[chunk[0]:chunk[1]]

            del A

            gc.collect()

            

        first_term_gradient = 2*torch.t(A1@A2 - A3@A4)

        

    elif type_sim == 'rbf':

        knots = list(torch.arange(0, N, step = batchSize).cuda())

        knots.append(N)

        batches = []

        for i in range(1,len(knots)):

            batches.append([knots[i-1],knots[i]])

        

        first_term_gradient = torch.zeros((d,1)).cuda().float()

        for batch in batches:

            D = rbf_similarity(X[batch[0]: batch[1]], X[batch[0]: batch[1]])

            A = D @ (torch.ones(((batch[1] - batch[0]),1)).cuda().float())

            B = D @ sig[batch[0]: batch[1]]

            first_term_gradient += (2 * (grad_sig[batch[0]: batch[1]] * X[batch[0]: batch[1]]).T @ 

                                        ((sig[batch[0]: batch[1]] * A) - B))

    return(2/(N**2) * first_term_gradient.flatten())
def second_term_gradient_torch(X, y, z, H, sig, theta):

    N = X.shape[0]

    d = X.shape[1]

    K = H.shape[1]



    # Reshape

    y = y.view(N,1)

    theta = theta.view(d,1)

    

    label = torch.sign(torch.sparse.mm(H.t(), y))



    grad_sig = sig * (1 - sig) * X

    

    return(torch.mean((H.t()@(z*sig) - label) * (H.t()@(z*grad_sig)), dim = 0))
# Testing the gradient functions

def first_grad_test(X, y, H, theta, type_sim = 'rbf'):

    N = X.shape[0]

    d = X.shape[1]

    K = H.shape[1]



    theta = theta.reshape(d,1)

    if type_sim == 'rbf':

        similarities = rbf_similarity_cpu(X, 0, N)

    elif type_sim == 'cosine':

        similarities = cosine_similarity(X)

    sig = logistic.cdf(X @ theta).reshape(N,1)

    grad_sig = sig * (1- sig) * X



    first_term = np.zeros(d)

    for i in range(N):

        for j in range(N):

            first_term += similarities[i,j] * (sig[i] - sig[j]) * (grad_sig[i] - grad_sig[j])

    first_term = 2 * first_term/(N**2)

    return(first_term)



def second_grad_test(X, y, H, theta):

    N = X.shape[0]

    d = X.shape[1]

    K = H.shape[1]



    theta = theta.reshape(d,1)

    sig = logistic.cdf(X@theta)

    grad_sig = sig * (1 - sig)

    second_term = np.zeros(d)

    true_label = np.sign(H.T @ y)

    idx = [0] + list(np.cumsum(np.sum(H.toarray(), axis = 0)))

    blocks = [[int(idx[i]), int(idx[i+1])] for i in range(len(idx)-1)]

    for i, block in enumerate(blocks):

        A = np.mean(sig[block[0]:block[1]], axis = 0)

        B = np.mean(grad_sig[block[0]:block[1]] * X[block[0]:block[1]], axis = 0)

        second_term += (A - true_label[i]) * B

    

    # aver_sig_doc = np.array(H.T @ sig) / np.sum(H, axis = 0).reshape(K,1)

    # second_term = np.mean((aver_sig_doc - true_label)**2)

    second_term = 1/K * second_term

    return(second_term)



print('Test functions')

print(first_grad_test(X, y, H, theta, type_sim = 'rbf'))

print(first_grad_test(X, y, H, theta, type_sim = 'cosine'))

print(second_grad_test(X, y, H, theta))



print('Used functions')

print(first_term_gradient_torch(X_gpu, sig, numChunk = 5, type_sim = 'rbf'))

print(first_term_gradient_torch(X_gpu, sig, numChunk = 5, type_sim = 'cosine'))

print(second_term_gradient_torch(X_gpu, y_gpu, z_gpu, H_gpu, sig, theta_gpu))
class MILoptimizer:

    def __init__(self, lamb = 10, alpha = 0.05, min_alpha = 0.005, momentum = 0.8, tol = 1e-4, numChunk = 5,

       init_theta = None, num_iter = 1000, random_state = 42, type_learn = 'const', batchSize = 2**11):

        import numpy as np

        import time

        

        from tqdm import tqdm_notebook as tqdm

        

        # Module for GPUs usage

        try:

            import torch

            torch.cuda.init()

        except:

            print('Package "torch" is not found')

        

        self.lamb = lamb

        self.alpha = alpha

        self.min_alpha = min_alpha

        self.tol = tol

        self.init_theta = init_theta

        self.num_iter = num_iter

        self.numChunk = numChunk

        self.random_state = random_state

        self.type_learn = type_learn

        self.batchSize = batchSize

        self.momentum = momentum

        

    def SGD_GPU(self, X, y, H, type_sim = 'cosine'):

        # Convert the data arrays into tensors

        X = torch.from_numpy(X).cuda().float()

        y = torch.from_numpy(y).cuda().float()

        

        # Convert D into tensor

        values = H.data

        indices = np.vstack((H.row, H.col))



        i = torch.LongTensor(indices)

        v = torch.FloatTensor(values)

        shape = H.shape

        

        H = torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()

        

        # Dimensions

        N = X.shape[0]

        d = X.shape[1]



        # Number of bags

        K = shape[1]



        # Compute z

        z = torch.sparse.mm(H, 1/torch.sparse.sum(H, dim = 0).to_dense().view(K,1)).view(N,1)

        

        # Start iterating

        it = 0



        # Initial theta

        if self.init_theta == None:

            theta_old = torch.randn(d).cuda()

        elif self.init_theta:

            theta_old = torch.from_numpy(self.init_theta).cuda()



        pbar = tqdm(total = self.num_iter)

        loss = torch.zeros(self.num_iter).cuda()

        V = torch.zeros(d).cuda().float()

        while it < self.num_iter:

            # Compute the gradient

            sigmoid = 1/(1 + torch.exp(-X @ (theta_old.view(d,1)))).view(N,1)

            

            second_term = self.lamb * second_term_gradient_torch(X, y, z, H, sigmoid, theta_old)

            

            first_term = first_term_gradient_torch(X, sigmoid, numChunk = self.numChunk, type_sim = type_sim, batchSize = self.batchSize)

            

            gradient = first_term.view(d) + second_term.view(d)

            if (int(it % 100) == 0 and it > 0) or it == (self.num_iter-1):

                print(torch.mean(torch.abs(gradient)))

            

            if self.type_learn == 'const':

                alpha = min(self.alpha, self.min_alpha)

            elif self.type_learn == 'diminish':

                alpha = min(self.alpha * (1 - it/self.num_iter), self.min_alpha)

            

            # Use the momentum

            '''V = self.momentum * V + (1 - self.momentum) * gradient

            theta_new = theta_old - alpha * V'''

            

            theta_new = theta_old - alpha * gradient

            

            sigmoid = 1/(1 + torch.exp(-X @ (theta_new.view(d,1)))).view(N,1)



            # Compute loss

            '''first_term_loss = first_term_torch(X, sigmoid, numChunk = self.numChunk, type_sim = type_sim, batchSize = self.batchSize)

            second_term_loss = self.lamb * second_term_torch(X, y, z, H, sigmoid, theta_new)

            loss[it] = first_term_loss + second_term_loss'''

            

            if torch.mean(torch.abs(gradient)) < self.tol:

                print('Optimal solution found after {} iterations!!!'.format(it+1))

                return(theta_new.cpu().numpy(), loss[:(it+1)].cpu().numpy(), gradient.cpu().numpy())

            else:

                theta_old = theta_new

                it += 1

                pbar.update(1)

        print('The solution has not converged, maximum iteration is reached!!!')

        pbar.close()

        return(theta_new.cpu().numpy(), loss.cpu().numpy(), gradient.cpu().numpy())

    

def predict(X, y, H, theta):

    from scipy.stats import logistic

    from sklearn.metrics import confusion_matrix



    N = X.shape[0]

    d = X.shape[1]

    K = H.shape[1]

        

    theta = theta.reshape(d,1)

    sentiment_sentences = logistic.cdf(X@theta) - 0.5

    pred_y = (np.sign(coo_matrix.transpose(H) @ np.sign(sentiment_sentences))/2 + 0.5).flatten()

    true_y = np.sign(coo_matrix.transpose(H) @ y)



    match = pred_y[pred_y == true_y]

    labels = np.unique(y)

    acc_label = np.zeros((len(labels),1))

    for label in labels:

        acc_label = len(match[match == label])/len(true_y == label)

    return(len(match)/K, acc_label, labels, pred_y)
class validation:

    def __init__(self, alpha = 0.05, tol = 1e-4,

                 init_theta = None, num_iter = 1000, random_state = 42, 

                 grid = np.arange(1, 16, step = 1), numChunk = 5):

        import numpy as np

        import time

        from sklearn.model_selection import KFold, train_test_split

        from tqdm import tqdm_notebook as tqdm

        

        # Module for GPUs usage

        try:

            import torch

            torch.cuda.init()

        except:

            print('Package "torch" is not found')

        

        self.alpha = alpha

        self.tol = tol

        self.init_theta = init_theta

        self.num_iter = num_iter

        self.random_state = random_state

        self.grid = grid

        self.numChunk = numChunk

        

    def fit_cross_validation(self, X, y, z, H, nfold = 5):

        # Split the dataset into KFold

        kf = KFold(n_splits = nfold)

        acc = np.zeros(len(self.grid))

        

        N = H.shape[0]

        K = H.shape[1]

        

        index_n = np.arange(0, N)

        for i, lambd in enumerate(list(self.grid)):

            s = time.time()

            # Split the data into train and test set

            acc_each_epoch = 0

            for doc_train_index, doc_test_index in kf.split(np.arange(0, K)):

                n_start = np.where(H.col == doc_test_index[0])[0][0]

                n_end = np.where(H.col == doc_test_index[-1])[0][-1]

                sent_test_index = np.arange(n_start, n_end + 1)

                sent_train_index = np.delete(index_n, sent_test_index)

                

                mil = MILoptimizer(lamb = lambd, num_iter = self.num_iter, numChunk = self.numChunk)

                theta_opt,_ = mil.SGD_GPU(X[sent_train_index], y[sent_train_index], z[sent_train_index], 

                                        H.tocsr()[sent_train_index][:, doc_train_index].tocoo())

                # Test

                acc_each_epoch += predict(X[sent_test_index], y[sent_test_index], 

                                          H.tocsr()[sent_test_index][:, doc_test_index].tocoo(), theta_opt)[0]

                

            acc[i] = acc_each_epoch/nfold

            e = time.time()

            np.save('accuracy_cv_topic.npy', acc[:i])

            print('Lambda: {}, Accuracy: {}, Time: {}'.format(lambd, acc[i], round(e-s,5)))

        return(acc)
def prediction_document(X_train, H_train, y_train, X_test, H_test, y_test, C = 1, max_iter = 100):

    # Dimensions

    N_train = X_train.shape[0]

    N_test = X_test.shape[0]

    d = X_train.shape[1]

    K_train = H_train.shape[1]

    K_test = H_test.shape[1]

    

    z_train = (H_train @ np.array(1/H_train.sum(axis = 0)).reshape(K_train,1)).reshape(N_train,1)

    z_test = (H_test @ np.array(1/H_test.sum(axis = 0)).reshape(K_test,1)).reshape(N_test,1)

    

    y_train = np.sign(H_train.T @ y_train)

    y_test = np.sign(H_test.T @ y_test)

    

    # Aggregate to the document level

    doc2vec_train = H_train.T @ (X_train * z_train)

    doc2vec_test = H_test.T @ (X_test * z_test)

    

    # Prediction

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C = C, max_iter = max_iter, random_state = 0).fit(doc2vec_train, y_train)

    y_pred_test = clf.predict(doc2vec_test)

    y_pred_train = clf.predict(doc2vec_train)

    

    from sklearn.metrics import confusion_matrix

    cm_train = confusion_matrix(y_train, y_pred_train)

    cm_test = confusion_matrix(y_test, y_pred_test)

    return(cm_train, np.sum(np.diag(cm_train))/np.sum(cm_train), 

           cm_test, np.sum(np.diag(cm_test))/np.sum(cm_test))
# Load the data

train = np.load('/kaggle/input/imdbmil/imdb_mil_train.npy', allow_pickle = True)

test = np.load('/kaggle/input/imdbmil/imdb_doc2vec_test_eachsent.npy', allow_pickle = True)



H_train = load_npz('/kaggle/input/imdbmil/H_train.npz')

H_test = coo_matrix(np.eye(test.shape[0]))
C = np.arange(2, 22, step = 2)

acc_out = np.zeros(len(C))

acc_in = np.zeros(len(C))

for i, c in enumerate(tqdm(C)):

    _,acc_in[i],_,acc_out[i] = prediction_document(train[:,:300], H_train, (np.sign(train[:,300]-5)+1)/2, 

                                        test[:,:300], H_test, test[:,300], C = c, max_iter = 1000)

plt.figure()

plt.plot(C, acc_out)

plt.plot(C, acc_in)

plt.show()
# Exclude the full-zero rows

# Training data

idx = ((train[:,:300]**2).sum(axis = 1) != 0).nonzero()[0]

train = train[idx]

H_train = H_train.tocsr()[idx,:]



# Testing data

idx = ((test[:,:300]**2).sum(axis = 1) != 0).nonzero()[0]

test = test[idx]
# Training

from sklearn.metrics import confusion_matrix

lambs = np.arange(50, 55, step = 5)

for lamb in lambs:

    obj = MILoptimizer(lamb = lamb, alpha = 0.05, tol = 1e-4, numChunk = 5, 

                       num_iter = 1000, type_learn = 'diminish', batchSize = 2**10)

    theta_opt, L, grad = obj.SGD_GPU(train[:,:300], (np.sign(train[:,300]-5)+1)/2, H_train.tocoo(), type_sim = 'rbf')

    # Visualize

    plt.plot(L)

    plt.show()

    print('---------------------')

    print('Gradient: \n', grad)

    print('---------------------')

    print('Average of elements in gradient: ', np.mean(np.abs(grad)))

    # Testing

    sent_sent = logistic.cdf(test[:,:300] @ theta_opt.reshape(300,1))

    pred_y = (np.sign(sent_sent - 0.5) + 1)/2

    cm = confusion_matrix(test[:,300], pred_y)

    print('---------------------')

    print('The confusion matrix is: ', cm)

    print('The out-of-sample accuracy is: ', np.sum(np.diag(cm))/np.sum(cm))