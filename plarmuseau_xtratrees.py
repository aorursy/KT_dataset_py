# Import the required libraries 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import json
import numpy as np # linear algebra

# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')

# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
	text_data = [", ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]


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


# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True,ngram_range=(1,2))
def tfidf_features(txt, flag):
    if flag == "train":
    	x = tfidf.fit_transform(txt)
    else:
	    x = tfidf.transform(txt)
    x = x.astype('float16')
    return x 

XT=tfidf_features(train_text+test_text, flag="train")
X = tfidf_features(train_text, flag="test")
X_test = tfidf_features(test_text, flag="test")
#print(X)
X=bm25_weight(coo_matrix(X))

X_test=bm25_weight(coo_matrix(X_test))

##Xw=coo_matrix(X.astype(np.float32)).todense()
##X_testw=coo_matrix(X_test.astype(np.float32)).todense()
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)

from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

    
def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    

names = [
         #'ElasticNet',
         #'SVC',
         #'kSVC',
         #'KNN',
         #'DecisionTree',
         'ExtraTree',
         # 99%'RandomForestClassifier',
         #'GridSearchCV',
         # 25% 'HuberRegressor',
         # 26%'Ridge',
         # 35% 'Lasso',
         # 26% 'LassoCV',
         # 39%'Lars',
         #'BayesianRidge',
         # 11% 'SGDClassifier',
         # 15 'RidgeClassifier',
         # 9 'LogisticRegression',
         #27 'OrthogonalMatchingPursuit',
         #'RANSACRegressor',
         ]

classifiers = [
    #ElasticNetCV(cv=10, random_state=0),
    #SVC(),
    #SVC(kernel = 'rbf', random_state = 0),
    #KNeighborsClassifier(n_neighbors = 10),
    #DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    # 99% RandomForestClassifier(n_estimators = 200),
    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),
    # HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),
    #Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),
    #Lasso(alpha=0.05),
    #LassoCV(),
    #Lars(n_nonzero_coefs=10),
    #BayesianRidge(),
    #SGDClassifier(),
    #RidgeClassifier(),
    #LogisticRegression(),
    #OrthogonalMatchingPursuit(),
    #RANSACRegressor(),
]
correction= [0,0,0,0,0,0,0,0,0,0,0,0]

temp=zip(names,classifiers,correction)
print(temp)

for name, clf,correct in temp:
    regr=clf.fit(X,y)
    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)
    print(name,'%error',procenterror(regr.predict(X),y),'rmsle',rmsle(regr.predict(X),y))
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score

    # Confusion Matrix
    print(name,'Confusion Matrix')
    print(confusion_matrix(y, np.round(regr.predict(X) ) ) )
    print('--'*40)

    # Classification Report
    print('Classification Report')
    print(classification_report(y,np.round( regr.predict(X) ) ))

    # Accuracy
    print('--'*40)
    logreg_accuracy = round(accuracy_score(y, np.round( regr.predict(X) ) ) * 100,2)
    print('Accuracy', logreg_accuracy,'%')
    
    # Predictions 
    print ("Predict on test data ... ")
    y_test = regr.predict(X_test)
    y_pred = lb.inverse_transform(y_test)
# Submission
print ("Generate Submission File ... ")
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
print(sub)
sub.to_csv('svm_output.csv', index=False)