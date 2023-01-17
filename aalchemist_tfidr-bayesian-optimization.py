import numpy as np # linear algebra
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import json
def read_dataset(path):
    return json.load(open(path))
from random import shuffle
train = read_dataset('../input/train.json')
shuffle(train)
test = read_dataset('../input/test.json')
def generate_text(data):
    text_data = [" ".join(doc['ingredients']).lower() for doc in data]
    return text_data 
train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]
# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    #x = x.astype('float16')
    return x 
X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")

X=X.toarray().tolist()
X_test = X_test.toarray().tolist()
lb = LabelEncoder()
y = lb.fit_transform(target)
'''from sklearn import decomposition
length1 = len(X)
length2=  len(X_test)
xlines=X[:]+X_test[:]
length3=len(xlines)
print(length1, length2, length3)
LENGTH= 500
pca = decomposition.PCA(n_components=LENGTH)
pca.fit(xlines)
xlines = pca.transform(xlines)
xlines=xlines.tolist()
print(len(xlines[0]))
X = xlines[0: length1]
X_test = xlines[length1:]
print(len(X), len(X_test))'''
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
'''def evaluate(gamma,C,degree ):
    classifier = SVC(C=C, # penalty parameter, setting it to a larger value 
                 kernel='rbf', # kernel type, rbf working fine here
                 degree=degree, # default value, not tuned yet
                 gamma=gamma, # kernel coefficient, not tuned yet
                 coef0=1, # change to 1 from default value of 0.0
                 shrinking=True, # using shrinking heuristics
                 tol=0.001, # stopping criterion tolerance 
                 probability=False, # no need to enable probability estimates
                 cache_size=200, # 200 MB cache size
                 class_weight='balanced', # all classes are treated equally 
                 verbose=0, # print the logs 
                 max_iter=10, # no limit, let it run
                 decision_function_shape=None, # will use one vs rest explicitly 
                 random_state=None)
    model = OneVsRestClassifier(classifier, n_jobs=-1)
    model.fit(X[0:33000], y[0:33000])
    return model.score(X[33000:],y[33000:])
    
m=BayesianOptimization(evaluate, { 
           #"kernel" : [ 'poly', 'rbf', 'sigmoid'],
           "gamma" : (1e-1 , 1e-4),
           "C" : (100, 1000),
    'degree':(2,4)
    
})'''
'''def evaluate(ESTIMATOR,DEAPTH,LEAF ,MNC):
    ESTIMATOR=int(ESTIMATOR)
    DEAPTH=int(DEAPTH)
    LEAF=int(LEAF)
    MNC=int(MNC)
    model = RandomForestClassifier(n_estimators=ESTIMATOR, criterion='gini',
                                   max_depth=DEAPTH,
                                   min_samples_split=MNC, 
                                    max_features="auto", max_leaf_nodes=LEAF,
                                   oob_score=True, n_jobs=-1, verbose=0)
    model.fit(X[0:33000], y[0:33000])
    ans= model.score(X[33000:],y[33000:])
    #model.fit(X[7000:], y[7000:])
    #ans+=model.score(X[:7000],y[:7000])
    del(model)
    return ans
m=BayesianOptimization(evaluate, { 
           #"kernel" : [ 'poly', 'rbf', 'sigmoid'],
           "ESTIMATOR" : (5 , 1000),
           "DEAPTH" : (100, 5000),
    'LEAF':(5,1000),
    'MNC':(2,100)
    
})



m.maximize(init_points=3, n_iter=10, acq='ei')'''

#params = m.res['max']['max_params']
#params['max_depth'] = float(params['max_depth'])
#print(params)
#'ESTIMATOR': 1000.0, 'DEAPTH': 1862.1660840628444, 'LEAF': 1000.0, 'MNC': 100.0}
'''classifier = SVC(C=1000, # penalty parameter, setting it to a larger value 
                 kernel='rbf', # kernel type, rbf working fine here
                 degree=3, # default value, not tuned yet
                 gamma=5, # kernel coefficient, not tuned yet
                 coef0=1, # change to 1 from default value of 0.0
                 shrinking=True, # using shrinking heuristics
                 tol=0.001, # stopping criterion tolerance 
                 probability=False, # no need to enable probability estimates
                 cache_size=200, # 200 MB cache size
                 class_weight='balanced', # all classes are treated equally 
                 verbose=False, # print the logs 
                 max_iter=-1, # no limit, let it run
                 decision_function_shape=None, # will use one vs rest explicitly 
                 random_state=None)
model = OneVsRestClassifier(classifier, n_jobs=-1)
model.fit(X, y)'''


model = RandomForestClassifier(n_estimators=2000, criterion='gini',
                                   max_depth=20000,
                                   min_samples_split=200, 
                                    max_features="auto", max_leaf_nodes=2000,
                                   oob_score=True, n_jobs=-1, verbose=1)
model.fit(X, y)





# Label Encoding - Target 
print ("Label Encode the Target Variable ... ")


# Model Training 
print ("Train the model ... ")


# Predictions 
print ("Predict on test data ... ")
y_test = model.predict(X_test)
y_pred = lb.inverse_transform(y_test)

# Submission
print ("Generate Submission File ... ")
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)

