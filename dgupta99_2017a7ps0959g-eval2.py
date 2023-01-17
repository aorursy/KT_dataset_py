#feature selection 

import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt



df = pd.read_csv('train.csv')

#creating train dataset x and y

arr = df.to_numpy()

x = np.delete(arr, [0,1,4,6,8,9, 10], axis = 1)

y= np.delete(arr, [0,1,2,3,4,5,6,7,8,9], axis = 1)



switcher = { 1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5} 



#for i in range(len(y)):

    #y[i] = switcher[float(y[i])]



#y = [int(i) for i in y]



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
import xgboost as xgb

for i in range(1,6):

    for j in range(3,11):

        clf = xgb.XGBClassifier(objective="multi:softprob", random_state=0, n_estimators = i*50, num_class = 6, max_depth = j, learning_rate = 0.1 )

        clf.fit(X_train, y_train)

        result = clf.predict(X_test)

        print(i*50," ",j,": ") 

        print(accuracy_score(y_test, result))
#xgboost





#clf = xgb.XBGClassifier(n_estimators = 100)



from sklearn.multiclass import OneVsRestClassifier



clf = xgb.XGBClassifier(objective="multi:softprob", random_state=42, n_estimators = 100, num_class = 6, max_depth = 10 , learning_rate = 0.1)
clf.fit(X_train,y_train)
result = clf.predict(X_test)
accuracy_score(y_test, result)
clf.fit(x,y)
test = pd.read_csv('test.csv')

test_arr = test.to_numpy()

test_arr = np.delete(test_arr, [0,1,4,6,8,9], axis = 1)
result_f = clf.predict(test_arr)
sub_fd_3 = pd.DataFrame()

sub_fd_3['id'] = test['id']

sub_fd_3['class'] = result_f



sub_fd_3.to_csv('submission_fd_3.csv', sep = ',')
x = pd.DataFrame(data = x)

y = pd.DataFrame(data = y)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#lightgbm

import lightgbm as lgb

train_data=lgb.Dataset(X_train,label=y_train)
y_train
param = {'num_leaves':150, 'objective':'multiclass','max_depth':7,'learning_rate':.05,'max_bin':200, 'num_class':6}

num_round=50



#lgbm=lgb.train(param,train_data,num_round)

#ypred2=lgbm.predict(X_test)



#print(accuracy_score(y_test, ypred2))



import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)

params = {

          "objective" : "multiclass",

          "num_class" : 6,

          "max_depth": -1 ,

          "learning_rate" : 0.1,

          "bagging_fraction" : 1,  # subsample

          "feature_fraction" : 1,  # colsample_bytree

          "bagging_freq" : 5,        # subsample_freq

          "bagging_seed" : 42,

          "verbosity" : -1 }

clf = lgb.train(params, d_train, 200)
result_f = clf.predict(X_test)

#print(accuracy_score(y_test, result_f))
print(result_f)
result_f = np.argmax(result_f, axis = 1)
result_f
reverse_switcher = {0:1, 1:2, 2: 3, 3: 5, 4: 6, 5:7}

result_f = [reverse_switcher[int(result_f[i])] for i in range(len(result_f))]
result_f
result_f
accuracy_score(y_test, result_f)
from lightgbm import LGBMClassifier



lgbm = LGBMClassifier(objective='multiclass', learning_rate = 0.5)



lgbm.fit(X_train, y_train)



y_pred = lgbm.predict(X_test)



accuracy_score(y_test, y_pred)



test = pd.read_csv('test.csv')

test_arr = test.to_numpy()

test_arr = np.delete(test_arr, [0,1,4,6,8,9], axis = 1)
result = lgbm.predict(test_arr)
sub_fd_4 = pd.DataFrame()

sub_fd_4['id'] = test['id']

sub_fd_4['class'] = result



sub_fd_4.to_csv('submission_fd_5.csv', sep = ',')
for i in range(1,6):

    for j in range(3,11):

        clf = LGBMClassifier(objective = 'multiclass', random_state=42, n_estimators = i*50, num_class = 6, max_depth = j,num_leaves = 750, min_child_samples = 3, bagging_fraction =0.4,  # subsample

          feature_fraction =1, 

          bagging_freq =5,       

          bagging_seed= 42)

        #clf = LGBMClassifier(objective="multiclass", random_state=0, n_estimators = i*50, num_class = 6, max_depth = j, learning_rate = 0.5 )

        clf.fit(X_train, y_train)

        result = clf.predict(X_test)

        print(i*50," ",j,": ") 

        print(accuracy_score(y_test, result))
result = clf.predict(test_arr)
sub_fd_4 = pd.DataFrame()

sub_fd_4['id'] = test['id']

sub_fd_4['class'] = result



sub_fd_4.to_csv('submission_fd_6.csv', sep = ',')