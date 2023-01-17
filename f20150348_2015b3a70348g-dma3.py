#from google.colab import drive



#drive.mount('/content/drive')
#cd drive/

#cd My Drive

#cd dataminingassignment3
ls


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv('../input/opcode_frequency_malware.csv')
data_orig.head()
data_orig2 =pd.read_csv('../input/opcode_frequency_benign.csv')
data_orig2.head()
from collections import Counter

listmalware=data_orig2.nunique()

listbenign=data_orig.nunique()
print(len(listbenign),len(listmalware))
differ=[]

for i in range(0,len(listbenign)):

  if(listmalware[i]!=listbenign[i]):

    differ.append(i)
print(len(differ))
gg=[]

for i in range(0,len(listbenign)):

  if i not in differ:

    gg.append(i)

    
type(data_orig2.columns[2])
data_orig['1809']=0

data_orig2['1809']=1
data_orig['1809'].value_counts()
data_orig.head()
#print(differ[0])
#print(listbenign[0],listmalware[0])
#data_orig.columns
#
# df=data_orig

# for i in range(1, len(df.columns)):

#     print(df.columns[i])

#     print(Counter(df.iloc[:,i]).keys()) # equals to list(set(words))

#     print(Counter(df.iloc[:,i]).values())# c
data=[data_orig,data_orig2]

df=pd.concat(data)
df.head()
df=df.drop(columns='FileName')
x=df.drop(columns='1809')
df.describe()
x.head()
y=df['1809']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from time import time

from scipy.stats import randint as sp_randint



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.datasets import load_digits

from sklearn.ensemble import RandomForestClassifier
# Clr=[0.001,0.01,0.1,1,10,100,1000,10000,100000,10000000,10000000]

# train_accuracy=[]

# test_accuracy=[]

# # for i in Clr:

#   clf = LogisticRegression(C=i).fit(x_train,y_train)

#   train_accuracy.append(accuracy_score(clf.predict(x_train),y_train))

#   test_accuracy.append(accuracy_score(clf.predict(x_test),y_test))

#   print("for C="+str(i))

#   print("training data accuracy:"+str(accuracy_score(clf.predict(x_train),y_train)))

#   print("validation accuracy:"+str(accuracy_score(clf.predict(x_test),y_test)))

#   print("==============================================")
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier
'''

from sklearn import metrics



def auc(m, train, test): 

    return (metrics.roc_auc_score(y_train,m.predict_proba(x_train)[:,1]),

                            metrics.roc_auc_score(y_test,m.predict_proba(x_test)[:,1]))



# Parameter Tuning

model = xgb.XGBClassifier()

param_dist = {"max_depth": [10,30,50,70,90],

              "min_child_weight" : [1,3,6,9,11],

              "n_estimators": [50,100,200,250,500],

              "learning_rate": [0.05, 0.1,0.16,0.2,0.25,0.3],}

grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 

                                   verbose=10, n_jobs=-1)

grid_search.fit(x_train, y_train)



grid_search.best_estimator_



#model = xgb.XGBClassifier(max_depth=50, min_child_weight=1,  n_estimators=200,\

#                          n_jobs=-1 , verbose=1,learning_rate=0.16)

#model.fit(x_train,y_train)



auc(model,x_train,x_test)

'''
'''

clf = RandomForestClassifier(n_estimators=200)





# Utility function to report best scores

def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")





# specify parameters and distributions to sample from

param_dist = {"max_depth": [3, None],

              "max_features": sp_randint(1, 11),

              "min_samples_split": sp_randint(2, 11),

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}



# run randomized search

n_iter_search = 20

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,

                                   n_iter=n_iter_search, cv=5)



start = time()

random_search.fit(x, y)

print("RandomizedSearchCV took %.2f seconds for %d candidates"

      " parameter settings." % ((time() - start), n_iter_search))

report(random_search.cv_results_)

'''
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


'''

gb = GradientBoostingClassifier(n_estimators=200, learning_rate = 0.5, max_features=22, max_depth = 2, random_state = 0)

gb.fit(x_train, y_train)

predictions = gb.predict(x_test)

'''
# y_scores_gb = gb.decision_function(x_test)

# fpr_gb, tpr_gb, _ = roc_curve(y_test, y_scores_gb)

# roc_auc_gb = auc(fpr_gb, tpr_gb)



# print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))
#https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db

'''

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=1, gamma=0, learning_rate=0.16, max_delta_step=0,

       max_depth=10, min_child_weight=3, missing=None, n_estimators=200,

       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1)

'''
# clf=RandomForestClassifier()

# param_grid = {"max_depth": [3,5,7,9,11 ,None],

#               "max_features": [3,7,10,25,37,'auto', 'sqrt', 'log2'],

#               "min_samples_split": [2, 3, 5,7,10],

#               "n_estimators":[10,20,50,75,100,125,150,200,250,450,750],

#               "bootstrap": [True, False],

#               "max_leaf_nodes":[10,25,50,75,128,192,256,348],

#               "criterion": ["gini", "entropy"]}



# # run grid search

# grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)

# start = time()

# grid_search.fit(x, y)



# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."

#       % (time() - start, len(grid_search.cv_results_['params'])))

# report(grid_search.cv_results_)
#print(grid_search.cv_results_)
# model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

#        colsample_bytree=1, gamma=0, learning_rate=0.16, max_delta_step=0,

#        max_depth=10, min_child_weight=3, missing=None, n_estimators=200,

#        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

#        silent=True, subsample=1)

# model=GradientBoostingClassifier(n_estimators=200, learning_rate = 0.5, max_features=22, max_depth = 2, random_state = 0)

model=RandomForestClassifier(n_estimators=200,bootstrap=True,criterion='entropy',min_samples_leaf=7,max_depth=9,random_state=97,n_jobs=-1)

model=RandomForestClassifier(n_estimators=200,bootstrap=False, criterion='entropy',max_depth= None, max_features= 2, min_samples_split=6)

model=RandomForestClassifier(n_estimators=20,bootstrap=False, criterion='entropy',max_depth= None, max_features= 25, min_samples_split=5)



#regr=RandomForestClassifier(n_estimators=200,bootstrap=False,class_weight='balanced',max_leaf_nodes=256,criterion='entropy',min_samples_leaf=47,max_depth=15,min_samples_split=4,random_state=97,n_jobs=-1)



model.fit(x,y)
#from sklearn.naive_bayes import GaussianNB as NB

#nb = NB()



#nb.fit(x,y)
test_data=pd.read_csv('Test_data.csv')

test_data.drop(columns='Unnamed: 1809',inplace=True)

test_data.drop(columns='FileName',inplace=True)

# make predictions for test data

test_data.head()
prediction=model.predict(test_data)


zero=0

one=0

for i in range(0,len(prediction)):

  if(prediction[i]==0):

    zero=zero+1

  else:

    one=one+1



print(zero,one)
#prediction=nb.predict(test_data)
test_data=pd.read_csv('Test_data.csv')

answer = {"FileName" : test_data["FileName"], "Class" : prediction}

ans = pd.DataFrame(answer, columns = ["FileName","Class"])

ans.to_csv("submission.csv", index = False)

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

  

  csv = df.to_csv(index=False)

  b64 = base64.b64encode(csv.encode())

  payload = b64.decode()

  html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

  html = html.format(payload=payload,title=title,filename=filename)

  return HTML(html)



create_download_link(ans)