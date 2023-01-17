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

from sklearn.preprocessing import normalize, scale

from sklearn.model_selection import KFold



# Modules for parallelizing the code

from joblib import Parallel, delayed

import multiprocessing

from tqdm.notebook import tqdm



# Module for clearing the memory

import gc



# Module for GPU computing

import torch

import torch.nn as nn

import torch.nn.functional as F
def rbf_similarity(W):

    # Dimensions

    V = W.shape[0]

    d = W.shape[1]

    

    F1 = (W**2).sum(dim = 1).view(V,1) @ torch.ones((1,V)).cuda()

    F2 = W @ W.T

    return(torch.exp(-(F1 + F1.T - 2 * F2)))
def rbf_similarity_cpu(W):

    # Dimensions

    V = W.shape[0]

    d = W.shape[1]

    

    F1 = (W**2).sum(axis = 1).reshape(V,1) @ np.ones((1,V))

    F2 = W @ W.T

    return(np.exp(-(F1 + F1.T - 2 * F2)))
# The gradient of loss function w.r.t. theta

def gradient_theta(X, W_new, label, theta):

    # X is the document-term matrix (K by V matrix)

    # W is the word vector matrix (V by d matrix)

    # label is the label vector, in this case, it is the cummulative abnormal return vector (K by 1 vector)

    # Gam is the rotation matrix, (d by d matrix)

    # theta is the vector to learn the sentiment (d by 1 matrix)

    

    # V is the size of our vocabulary

    # K is the number of documents

    # d is the cardinality of word vector

    

    K = X.shape[0]

    V = X.shape[1]

    d = W_new.shape[1]

    

    # Reshape

    theta = theta.view(d,1)

    label = label.view(K,1)

    

    sigmoid = 1/(1 + torch.exp(-torch.sparse.mm(X, W_new) @ theta)).view(K,1)

      

    # Compute A

    A = (sigmoid - label).view(K,1)

    

    # Compute B

    grad_sig = sigmoid * (1 - sigmoid)

    

    A = (A * grad_sig * X.to_dense()).mean(dim = 0).view(V,1)

    

    grad_theta = W_new.T @ A.view(V,1)

    

    return(grad_theta.flatten())   # '''
# The gradient of the loss function w.r.t. Gamma

def gradient_W(X, W, W_new, label, theta, lamb = 0.01):

    # X is the document-term matrix (K by V matrix)

    # W is the word vector matrix (V by d matrix)

    # label is the label vector, in this case, it is the cummulative abnormal return vector (K by 1 vector)

    # Gam is the rotation matrix, (d by d matrix)

    # theta is the vector to learn the sentiment (d by 1 matrix)

    

    # V is the size of our vocabulary

    # K is the number of documents

    # d is the cardinality of word vector

    

    K = X.shape[0]

    V = X.shape[1]

    d = W.shape[1]

    

    # Reshape

    theta = theta.view(d,1)

    label = label.view(K,1)

    

    # First term

    first_term = 2 / (V*d) * (W_new - W)

    

    # Second term

    sigmoid = 1/(1 + torch.exp(-torch.sparse.mm(X, W_new) @ theta)).view(K,1)

    

    # Compute A

    A = (sigmoid  - label).view(K,1)

    

    # Compute B

    grad_sig = sigmoid * (1 - sigmoid)

    A = (A * grad_sig * X.to_dense()).mean(dim = 0).view(V,1)

    

    second_term = A @ theta.T

    

    return(first_term + lamb * second_term)   #  + gam * regu_term'''
def loss(X, W, W_new, label, theta, lamb = 0.01):

    # X is the document-term matrix (K by V matrix)

    # W is the word vector matrix (V by d matrix)

    # label is the label vector, in this case, it is the cummulative abnormal return vector (K by 1 vector)

    # Gam is the rotation matrix, (d by d matrix)

    # theta is the vector to learn the sentiment (d by 1 matrix)

    

    # V is the size of our vocabulary

    # K is the number of documents

    # d is the cardinality of word vector

    

    K = X.shape[0]

    V = X.shape[1]

    d = W.shape[1]

    

    # Reshape

    theta = theta.view(d,1)

    label = label.view(K,1)

    

    first_term = torch.mean((W_new - W)**2)

    

    # Second term

    sigmoid = 1/(1 + torch.exp(-torch.sparse.mm(X, W_new) @ theta)).view(K,1)

    

    second_term = torch.mean((sigmoid  - label)**2)

    

    return(first_term + lamb * second_term)  #  + gam * regu_term'''
class word2vec_sentiment:

    def __init__(self, alpha = 0.05, tol = 1e-4, lamb = 10,

       init_theta = None, init_W = None, num_iter = 1000, type_learn = 'const'):

        import numpy as np

        import time

        

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

        self.init_W = init_W

        self.lamb = lamb

        self.num_iter = num_iter

        self.type_learn = type_learn

        

    def estimate(self, X, W, label):

        # Convert X into tensor

        X = coo_matrix(X)



        values = X.data

        indices = np.vstack((X.row, X.col))



        i = torch.LongTensor(indices)

        v = torch.FloatTensor(values)

        shape = X.shape



        X = torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()

        

        # Transfer data from CPU to GPU

        # W = normalize(W)

        W = torch.from_numpy(W).cuda().float()

        label = torch.from_numpy(label).cuda().float()

        

        # Dimension

        K = X.shape[0]

        V = W.shape[0]

        d = W.shape[1]

        

        # Start iterating

        it = 0



        # Initial theta

        if self.init_theta == None:

            theta_old = torch.randn(d).cuda()

        elif self.init_theta:

            theta_old = torch.from_numpy(self.init_theta).cuda().float()

            

        # Initial W

        if self.init_W == None:

            W_tilde_old = W

        elif self.init_W:

            W_tilde_old = torch.from_numpy(self.init_W).cuda().float()

            

        pbar = tqdm(total = self.num_iter)

        L = torch.zeros(self.num_iter).cuda()

        while it < self.num_iter:

            # Learning rate type

            if self.type_learn == 'const':

                alpha = self.alpha

            elif self.type_learn == 'diminish':

                alpha = self.alpha * (1 - it/self.num_iter)

                

            # Compute the gradient w.r.t. theta

            grad_theta = gradient_theta(X, W_tilde_old, label, theta_old)

            

            # Compute the gradient w.r.t. Gamma

            grad_W = gradient_W(X, W, W_tilde_old, label, theta_old, lamb = self.lamb)

            

            # Update theta

            theta_new = theta_old - alpha * grad_theta

            

            # Update W

            W_tilde_new = W_tilde_old - alpha * grad_W

            

            # Normalize

            # W_tilde_new = F.normalize(W_tilde_new, dim = 1, p = 2)

            

            # Compute the loss

            L[it] = loss(X, W, W_tilde_new, label, theta_old, lamb = self.lamb)

            if (torch.mean(torch.abs(theta_new - theta_old)) < self.tol 

                and torch.mean(torch.abs(W_tilde_new - W_tilde_old)) < self.tol):

                print('Optimal solution found after {} iterations!!!'.format(it))

                return(theta_new.cpu().numpy(), W_tilde_new.cpu().numpy(), L[:(it+1)].cpu().numpy())

            else:

                theta_old = theta_new

                W_tilde_old = W_tilde_new

                it += 1

                pbar.update(1)

        print('The solution has not converged, maximum iteration is reached!!!')

        pbar.close()

        return(theta_new.cpu().numpy(), W_tilde_new.cpu().numpy(), L.cpu().numpy())
# Make prediction

def prediction(doc_term, word_vector, theta, true_label, discrete = True):

    # Dimensions

    K = doc_term.shape[0]

    V = doc_term.shape[1]

    d = word_vector.shape[1]

    

    # Reshape

    theta = theta.reshape(d,1)

    

    # Compute the probability of each word

    sig = logistic.cdf(word_vector @ theta)

    

    # Now, aggregate words to the document level

    # Firstly, convert the document-term matrix into the matrix with tf-idf weights

    idf = np.log(doc_term.shape[0]/doc_term.tocsr().getnnz(axis = 0))

    tf = csr_matrix(doc_term/doc_term.sum(axis = 1))

    tf_idf = csr_matrix(tf@csr_matrix(np.diag(idf)))

    

    # Aggregate words to documents

    normalization_term = np.array(tf_idf.sum(axis = 1)).reshape(K,1)

    

    # The sentiment score

    sentiment_score = (tf_idf @ sig)/normalization_term

    pred_label = (np.sign(np.sign(sentiment_score - 0.5) + 0.01) + 1)/2

    

    from sklearn.metrics import confusion_matrix

    if discrete == True:

        cm = confusion_matrix(true_label.flatten(), pred_label.flatten())

        return(pred_label, cm)

    elif discrete == False:

        return(pred_label, true_label)
def predict_document_level(doc_term_train, true_label_train, doc_term_test, true_label_test, 

                           new_word2vec_train, old_word2vec_test, common_words, discrete = True, 

                           type_classifier = 'svm', type_weight = 'raw', tuning = 10):

    # Dimensions

    K = doc_term_train.shape[0]

    V = doc_term_train.shape[1]

    d = word_vector_train.shape[1]

    

    old_word2vec_test.loc[common_words] = new_word2vec_train.loc[common_words]

    new_word2vec_train = new_word2vec_train.values

    new_word2vec_test = old_word2vec_test.values

    

    # Now, aggregate words to the document level

    if type_weight == 'tf_idf':

        # Firstly, convert the document-term matrix into the matrix with tf-idf weights

        idf_train = np.log(K/doc_term_train.tocsr().getnnz(axis = 0))

        tf_train = csr_matrix(doc_term_train/doc_term_train.sum(axis = 1))

        tf_idf_train = csr_matrix(tf_train@csr_matrix(np.diag(idf_train)))



        # Firstly, convert the document-term matrix into the matrix with tf-idf weights

        idf_test = np.log(doc_term_test.shape[0]/doc_term_test.tocsr().getnnz(axis = 0))

        tf_test = csr_matrix(doc_term_test/doc_term_test.sum(axis = 1))

        tf_idf_test = csr_matrix(tf_test@csr_matrix(np.diag(idf_test)))

        

        # Document vectors

        doc_vector_train = tf_idf_train @ new_word2vec_train

        doc_vector_test = tf_idf_test @ new_word2vec_test

    elif type_weight == 'raw':

        # Document vectors

        doc_vector_train = doc_term_train @ new_word2vec_train

        doc_vector_test = doc_term_test @ new_word2vec_test



    if type_classifier == 'svm':

        from sklearn.svm import SVC

        clf = SVC(gamma = 'auto')

        clf.fit(doc_vector_train, true_label_train)

    elif type_classifier == 'logistic':

        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C = tuning, max_iter = 10000, random_state = 0).fit(doc_vector_train, true_label_train)

    elif type_classifier == 'random_forest':

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(max_depth = 7, random_state = 0).fit(doc_vector_train, true_label_train)

    elif type_classifier == 'neural_network':

        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(hidden_layer_sizes = 500).fit(doc_vector_train, true_label_train)



    pred_label_train = clf.predict(doc_vector_train)

    pred_label_test = clf.predict(doc_vector_test)

    

    from sklearn.metrics import confusion_matrix

    if discrete == True:

        cm_train = confusion_matrix(true_label_train.flatten(), pred_label_train.flatten())

        cm_test = confusion_matrix(true_label_test.flatten(), pred_label_test.flatten())

        return(pred_label_train, cm_train, pred_label_test, cm_test)

    elif discrete == False:

        return(pred_label_train, true_label_train, pred_label_test, true_label_test)
class validation:

    def __init__(self,alpha = 0.05, tol = 1e-4, init_Gam = None, grid = np.array([0.1, 0.5, 1, 5, 10]),

       init_theta = None, num_iter = 1000, type_learn = 'diminish'):

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

        self.init_Gam = init_Gam

        self.num_iter = num_iter

        self.grid = grid

        self.type_learn = type_learn

        

    def fit_cross_validation(self, X, W, label, tuning = 10, nfold = 5, type_classifier = 'logistic', type_weight = 'raw'):

        # Split the dataset into KFold

        kf = KFold(n_splits = nfold)

        acc = np.zeros(len(self.grid))

        

        wordlist_train = W.index

        

        V = W.shape[0]

        K = X.shape[0]

        d = W.shape[1]

        

        A, B = np.min(label), np.max(label)

        for i, lamb in enumerate(list(self.grid)):

            s = time.time()

            # Split the data into train and test set

            acc_each_epoch = 0

            for doc_train_index, doc_test_index in kf.split(np.arange(0, K)):

                obj = word2vec_sentiment(num_iter = self.num_iter, tol = self.tol, 

                                         type_learn = self.type_learn, lamb = lamb)

                _, W_n, _ = obj.estimate(X.tocsr()[doc_train_index], W.values, 

                                             ((label - A)/(B - A))[doc_train_index])

                W_n = pd.DataFrame(W_n)

                W_n.index = wordlist_train

                

                # Test

                _, _, _, cm_test = predict_document_level(X.tocsr()[doc_train_index], (np.sign(label[doc_train_index] - 2.5) + 1)/2, 

                                                         X.tocsr()[doc_test_index], (np.sign(label[doc_test_index] - 2.5) + 1)/2, 

                                                         W_n, W_n, wordlist_train, type_classifier = type_classifier, 

                                                         type_weight = type_weight, tuning = tuning)

                acc_each_epoch += np.sum(np.diag(cm_test))/np.sum(cm_test)

            acc[i] = acc_each_epoch/nfold*100

            e = time.time()

            print('Lambda: {}, Accuracy: {}, Time: {}'.format(lamb, acc[i], round(e-s,5)))

        return(acc)
# Import the data

doc_term_train = load_npz('/kaggle/input/imdb-w2v-sentiment/doc_term_yelp_train.npz')

word_vector_train = pd.DataFrame(pd.read_csv('/kaggle/input/imdb-w2v-sentiment/word_matrix_yelp_train.csv', index_col = 0))

score_train = np.load('/kaggle/input/imdb-w2v-sentiment/score_yelp_train.npy')

word_matrix_train = word_vector_train.values

word_list_train = word_vector_train.index



# Import the testing data

doc_term_test = load_npz('/kaggle/input/imdb-w2v-sentiment/doc_term_yelp_test.npz')

word_vector_test = pd.DataFrame(pd.read_csv('/kaggle/input/imdb-w2v-sentiment/word_matrix_yelp_test.csv', index_col = 0))

score_test = np.load('/kaggle/input/imdb-w2v-sentiment/score_yelp_test.npy')

word_matrix_test = word_vector_test.values

word_list_test = word_vector_test.index



common_words = sorted(list(set(word_list_test) & set(word_list_train)))



doc_term_train = normalize(doc_term_train, norm = 'l1', axis = 1)

doc_term_test = normalize(doc_term_test, norm = 'l1', axis = 1)
# Word list

words = list(word_vector_train.index)

old_similarities = cosine_similarity(word_matrix_train)

def most_similar(word, wordlist, sim_matrix, num_word = 20, threshold = 0.708):

    # The threshold is proposed by Rekabsaz, Lupu and Hanbury (2017)

    if word not in wordlist:

        print('The word is not in your word list!!!')

        return([])

    else:

        col = sim_matrix[:,wordlist.index(word)]

        index_threshold = np.where(col >= threshold)[0]

        index_threshold = index_threshold[np.argsort(col[index_threshold])][::-1]

        index_num = col.argsort()[-num_word:][::-1]

        num_index = min(len(index_threshold), len(index_num))

        if num_index == len(index_threshold):

            index = index_threshold

        else:

            index = index_num

        return([[wordlist[i], np.round(sim_matrix[i,wordlist.index(word)], 3)] for i in index])
# Check the most similar words

chosen_word = 'good'

num_related_words = 10

print('Before rotation')

print(most_similar(chosen_word, words, old_similarities, num_related_words, threshold = 0.5))
print('With raw frequency weighting')

pred_label_train, cm_train, pred_label_test, cm_test = predict_document_level(doc_term_train, (np.sign(score_train - 2.5) + 1)/2, 

                                                                              doc_term_test, (np.sign(score_test - 2.5) + 1)/2, 

                                                                              word_vector_train, word_vector_test, common_words,

                                                                              type_classifier = 'logistic', type_weight = 'raw')

print('The confusion matrix in-sample is {}, and out-of-sample is {}.'.format(cm_train, cm_test))

print('The in-sample accuracy is {} and out-of-sample accuracy is {}.'.format(np.sum(np.diag(cm_train))/np.sum(cm_train)*100, 

                                                                             np.sum(np.diag(cm_test))/np.sum(cm_test)*100))
idf_train = np.log(doc_term_train.shape[0]/doc_term_train.tocsr().getnnz(axis = 0))

tf_train = csr_matrix(doc_term_train/doc_term_train.sum(axis = 1))

tf_idf_train = csr_matrix(tf_train @ csr_matrix(np.diag(idf_train)))
# Validation

hyper_lamb = np.exp(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

nfold = 3



# Shuffle the data

idx = np.arange(doc_term_train.shape[0])

np.random.seed(1)

np.random.shuffle(idx)



cv = validation(num_iter = 1000, tol = 1e-5, grid = hyper_lamb)

acc = cv.fit_cross_validation(doc_term_train[idx], word_vector_train, score_train[idx], nfold = nfold, tuning = 10)



# Visualization

fig = plt.figure()

plt.figure()

plt.plot(hyper_lamb, acc)

plt.xlabel('Tuning paramter: lambda')

plt.ylabel('Accuracy')

plt.title('Cross Validation, ' + str(nfold) + ' folds')

plt.show()
A, B = np.min(score_train), np.max(score_train)

lamb = float(hyper_lamb[acc == max(acc)])

obj = word2vec_sentiment(num_iter = 1000, tol = 1e-5, type_learn = 'diminish', lamb = lamb)

t, W_new, l = obj.estimate(doc_term_train, word_vector_train.values, (score_train - A)/(B - A))

W_new = pd.DataFrame(W_new)

W_new.index = word_list_train

tune = np.array([1, 5, 10, 20, 50, 100])



plt.plot(l)

plt.show()



for tuning in tune: 

    pred_label_train, cm_train, pred_label_test, cm_test = predict_document_level(doc_term_train, (np.sign(score_train - 2.5) + 1)/2, 

                                                                                  doc_term_test, (np.sign(score_test - 2.5) + 1)/2, 

                                                                                  W_new, word_vector_test, common_words, tuning = tuning,

                                                                                  type_classifier = 'logistic', type_weight = 'raw')

    



    print('For tuning parameter is {} \n The confusion matrix in-sample is {}, and out-of-sample is {}.'.format(tuning, cm_train, cm_test))

    print('The in-sample accuracy is {} and out-of-sample accuracy is {}'.format(np.sum(np.diag(cm_train))/np.sum(cm_train)*100, 

                                                                             np.sum(np.diag(cm_test))/np.sum(cm_test)*100))
# New word2vec representation

new_word2vec = W_new

similarities = cosine_similarity(new_word2vec)
np.save('new_word2vec_SWESA_yelp.npy', new_word2vec)

np.save('acc_yelp.npy', acc)
# Check the most similar words

chosen_word = 'good'

num_related_words = 11

print('After rotation')

print(most_similar(chosen_word, words, similarities, num_related_words, threshold = 0.5))

print('Before rotation')

print(most_similar(chosen_word, words, old_similarities, num_related_words, threshold = 0.5))
# List of word incorporated with sentiment

d = 300

sentiment_words = logistic.cdf(new_word2vec @ t.reshape(d,1))

label_words = dict([[words[i], sentiment_words[i]] for i in range(len(words))])



sentiment_words = np.sign(sentiment_words - 0.5)

# Number of negative words

print('Number of negative and positive words are {} and {} correspondingly'.format(sum(sentiment_words < 0), sum(sentiment_words >= 0)))

print('-----------------------------------------')



# Most positive words and most negative words

# Sort the cummulative abnormal returns

import operator

sorted_words_sent = sorted(label_words.items(), key = operator.itemgetter(1))



num_chosen_words = 200

print('The {} most negative words: '.format(num_chosen_words))

print(sorted_words_sent[:num_chosen_words])

print('-----------------------------------------')

print('The {} most positive words: '.format(num_chosen_words))

print(sorted_words_sent[-num_chosen_words:][::-1])
sorted_words_sent
sorted_words_sent[::-1]