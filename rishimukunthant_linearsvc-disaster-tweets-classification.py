import numpy as np

import pandas as pd
train_df = pd.read_csv('../input/nlp-getting-started/train.csv') 

test_df =  pd.read_csv('../input/nlp-getting-started/test.csv')
train_df
train_df[train_df['target']==0]['text'].values[0]
#Count Vectorizer class present in feature_extraction.text module from scikit learn so importing.

from sklearn import feature_extraction
#Python OOP so create a object as instance of CountVectorizer Class

tfidf_vec = feature_extraction.text.TfidfVectorizer()
#transform ur raw text to DTM using cnt_vec object so u can feed it to any ML model

#fit transform on train and transform on test. Followed convention. To prevent data leakage etc

dtm_train = tfidf_vec.fit_transform(train_df['text'])

dtm_test = tfidf_vec.transform(test_df['text'])
#from sklearn import linear_model

from sklearn import svm

from sklearn.linear_model import LogisticRegression

#from sklearn.ensemble import RandomForestClassifier
#Model object

#clf = svm.SVC(kernel='linear',C=1,gamma=1)

clf = LogisticRegression()

#clf = RandomForestClassifier()
# #Doing Grid Search

# from sklearn.model_selection import GridSearchCV 

  

# # defining parameter range 

# param_grid = {'C': [0.1, 1, 10, 100, 1000],  

#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

#               'kernel': ['rbf','linear']}  

  

# grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 0) 

  

# # fitting the model for grid search 

# grid.fit(dtm_train, train_df['target']) 



# # print best parameter after tuning 

# print(grid.best_params_) 

  

# # print how our model looks after hyper-parameter tuning 

# print(grid.best_estimator_) 

#need model_selection module to implement cross validation

from sklearn import model_selection
import statistics

#Getting scores

scores = model_selection.cross_val_score(clf, dtm_train, train_df['target'], cv=5, scoring='f1')

print(scores)

print(statistics.mean(scores))
#now train on complete dtm_train

clf.fit(dtm_train, train_df['target'])
#Now predict and store the pred in sample_submission and submit fresh submission

sample_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

#overwrite

sample_submission['target'] = clf.predict(dtm_test)

#submit csv file with just id and predictions without index

sample_submission.to_csv('LogisticRegressionsubmission.csv',index=False)