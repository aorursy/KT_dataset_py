import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#forked to learn the 'input

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.neural_network import MLPClassifier

import h5py

from scipy import sparse

import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

print("Modules imported!")

print("Collecting Data...")

hf = h5py.File("../input/cdk2.h5", "r")

ids = hf["chembl_id"].value # the name of each molecules

ap = sparse.csr_matrix((hf["ap"]["data"], hf["ap"]["indices"], hf["ap"]["indptr"]), shape=[len(hf["ap"]["indptr"]) - 1, 2039])

mg = sparse.csr_matrix((hf["mg"]["data"], hf["mg"]["indices"], hf["mg"]["indptr"]), shape=[len(hf["mg"]["indptr"]) - 1, 2039])

tt = sparse.csr_matrix((hf["tt"]["data"], hf["tt"]["indices"], hf["tt"]["indptr"]), shape=[len(hf["tt"]["indptr"]) - 1, 2039])

features = sparse.hstack([ap, mg, tt]).toarray() # the samples' features, each row is a sample, and each sample has 3*2039 features

labels = hf["label"].value # the label of each molecule

features
labels
from sklearn.preprocessing import normalize

from scipy.sparse import coo_matrix, csr_matrix



def cosine(plays):

    normalized = normalize(plays)

    return normalized.dot(normalized.T)





def bhattacharya(plays):

    plays.data = np.sqrt(plays.data)

    return cosine(plays)





def ochiai(plays):

    plays = csr_matrix(plays)

    plays.data = np.ones(len(plays.data))

    return cosine(plays)





def bm25_weight(data, K1=1.2, B=0.8):

    """ Weighs each row of the matrix data by BM25 weighting """

    # calculate idf per term (user)

    N = float(data.shape[0])

    idf = np.log(N / (1 + np.bincount(data.col)))



    # calculate length_norm per document (artist)

    row_sums = np.squeeze(np.asarray(data.sum(1)))

    average_length = row_sums.sum() / N

    length_norm = (1.0 - B) + B * row_sums / average_length



    # weight matrix rows by bm25

    ret = coo_matrix(data)

    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]

    return ret





def bm25(plays):

    plays = bm25_weight(plays)

    return plays.dot(plays.T)



def get_largest(row, N=10):

    if N >= row.nnz:

        best = zip(row.data, row.indices)

    else:

        ind = np.argpartition(row.data, -N)[-N:]

        best = zip(row.data[ind], row.indices[ind])

    return sorted(best, reverse=True)





def calculate_similar_artists(similarity, artists, artistid):

    neighbours = similarity[artistid]

    top = get_largest(neighbours)

    return [(artists[other], score, i) for i, (score, other) in enumerate(top)]







similarity = bm25(coo_matrix(features)).todense()



similarity
U, sigma, Vt = np.linalg.svd(similarity[:,:200], full_matrices=False)

sigma = np.diag(sigma)

print(U.shape,sigma.shape,Vt.shape)


from sklearn.model_selection import cross_val_predict

from sklearn import linear_model

import matplotlib.pyplot as plt



lr = linear_model.LinearRegression()

predicted = cross_val_predict(lr, U, labels, cv=5)



fig, ax = plt.subplots()

ax.scatter(labels, predicted, edgecolors=(0, 0, 0))

ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=1)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()



from sklearn.metrics import r2_score

print(r2_score(labels, predicted))
from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.preprocessing import MinMaxScaler



# import some data to play with

       # those ? converted to NAN are bothering me abit...        



from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}



X = U

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    

Y=labels





names = [

         'ElasticNet',

         'SVC',

         'kSVC',

         'KNN',

         'DecisionTree',

         'RandomForestClassifier',

         'GridSearchCV',

         'HuberRegressor',

         'Ridge',

         'Lasso',

         'LassoCV',

         'Lars',

         'BayesianRidge',

         'SGDClassifier',

         'RidgeClassifier',

         'LogisticRegression',

         'OrthogonalMatchingPursuit',

         #'RANSACRegressor',

         ]



classifiers = [

    ElasticNetCV(cv=10, random_state=0),

    SVC(),

    SVC(kernel = 'rbf', random_state = 0),

    KNeighborsClassifier(n_neighbors = 1),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators = 200),

    GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),

    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),

    Lasso(alpha=0.05),

    LassoCV(),

    Lars(n_nonzero_coefs=10),

    BayesianRidge(),

    SGDClassifier(),

    RidgeClassifier(),

    LogisticRegression(),

    OrthogonalMatchingPursuit(),

    #RANSACRegressor(),

]

correction= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

print(temp)



for name, clf,correct in temp:

    regr=clf.fit(X,Y)

    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)

    print(name,'%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score



    # Confusion Matrix

    print(name,'Confusion Matrix')

    print(confusion_matrix(Y, np.round(regr.predict(X) ) ) )

    print('--'*40)



    # Classification Report

    print('Classification Report')

    print(classification_report(Y,np.round( regr.predict(X) ) ))



    # Accuracy

    print('--'*40)

    logreg_accuracy = round(accuracy_score(Y, np.round( regr.predict(X) ) ) * 100,2)

    print('Accuracy', logreg_accuracy,'%')