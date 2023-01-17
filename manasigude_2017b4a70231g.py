import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



%matplotlib inline
df = pd.read_csv("train.csv")
data = df 
df.head()
zeros = df[df["target"]==0]
ones = df[df["target"]==1]
print(len(ones))
zeros = zeros.sample(n=2000,replace = True,random_state = 1)
print(len(zeros))
data_train = pd.concat([zeros,ones],axis = 0)
print(len(data_train))
data_train.drop(["id"],axis = 1,inplace = True)
Y = data_train["target"]

X = data_train.drop(["target"],axis = 1)

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()              

scaled_X_train = scaler.fit_transform(X) 

scaled_X_train
from xgboost import XGBClassifier
xgb = XGBClassifier(eta=0.01,objective='multi:softmax',num_class=2,max_depth=4,min_child_weight=1)

xgb.fit(scaled_X_train, Y)



#Tunning max_depth with range (3,4,7,9)

#Tunning min_depth with range (1,3,5)

#Tunning eta with (0.01,0.2,0.3)
from sklearn.model_selection import GridSearchCV
parameters = {'gamma':[i/10.0 for i in range(0,5)]}



xgb_cv = XGBClassifier()



clf = GridSearchCV(xgb_cv, parameters, verbose=1)



clf.fit(scaled_X_train, Y)
clf.best_params_
df_test = pd.read_csv("test.csv")

df_test.head()
id_arr = df_test["id"]

df_test.drop(["id"],axis = 1,inplace = True)

scaled_X_test = scaler.fit_transform(df_test) 

scaled_X_test
y_pred = clf.predict(scaled_X_test)
my_submission = pd.DataFrame({'id': id_arr,'target':y_pred})

my_submission.to_csv('submission2.8.csv',index = False)