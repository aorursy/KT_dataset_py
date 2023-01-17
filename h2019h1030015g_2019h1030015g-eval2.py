import pandas as pd

import numpy as np

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
df.isnull().sum()
df.astype(bool).sum(axis=0)
nonzeromean = df[df["chem_1"]!=0]["chem_1"].mean()

df[df["chem_1"]==0]["chem_1"]=nonzeromean

df.loc[df["chem_1"] == 0, 'chem_1'] = nonzeromean



X = df[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","chem_7","attribute"]]

y = df["class"]



df.astype(bool).sum(axis=0)
#That's very little and skewd data. :(

a = set(df["class"])

b = list(a)

bcoz = list(df["class"])

for i in range(0,6):

    print("Number of",b[i]," is ",bcoz.count(b[i]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
# Reduces accuracy. Strange :(

#scaler = StandardScaler()

#scaler.fit(X_train)



#X_train = scaler.transform(X_train)

#X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(metric = "manhattan",n_neighbors=3, weights = "distance")

classifier.fit(X_train, y_train)

y_pred2 = classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))
#grid_params = {

#    'n_neighbors' : [3,5,11,19],

#    'weights' : ['uniform','distance'],

#    'metric' : ['euclidean','manhattan']

#}





#grid_search = GridSearchCV(estimator = classifier,

#                           param_grid = grid_params,

#                           verbose=1,

#                           cv = 3,

#                           n_jobs = -1)

#grid_search = grid_search.fit(X_train, y_train)
#accuracy = grid_search.best_score_

#accuracy
#grid_search.best_params_
kf = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

kf.isnull().sum()

myid1 = kf["id"].copy()

X = kf[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","chem_7","attribute"]]



y = classifier.predict(X)



boo1 = pd.DataFrame(y)

ans = pd.concat([myid1,boo1],axis=1)

ans.columns=["id","class"]

ans = ans.set_index("id")

ans.to_csv("ans_knn.csv")
model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
kf = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

kf.isnull().sum()

myid1 = kf["id"].copy()

X = kf[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","chem_7","attribute"]]



y = model.predict(X)



boo1 = pd.DataFrame(y)

ans = pd.concat([myid1,boo1],axis=1)

ans.columns=["id","class"]

ans = ans.set_index("id")

ans.to_csv("ans_GBX.csv")
#grid_param = {

#    'n_estimators': [5,8,10,15,16,17,18,20,100,150,200],

#    'max_depth': [1,2,3,4,5,6,7,8,9,12],

#    'random_state': [0,1,2],

#    'objective':['binary:logistic'],

#    'learning_rate': [0.15], #so called `eta` value

#    'min_child_weight': [3,11],

#    'subsample': [0.9],

#    'colsample_bytree': [0.5],

#     'base_score' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7],



#}

    





#grid_search = GridSearchCV(estimator = model,

#                           param_grid = grid_param,

#                           verbose=1,

#                           cv = 3,

#                           n_jobs = -1)



#grid_search = grid_search.fit(X_train, y_train)
#accuracy = grid_search.best_score_

#accuracy
#grid_search.best_params_
rfmodel = RandomForestClassifier(criterion= 'entropy',max_depth= 3,min_samples_leaf= 2,min_samples_split= 2,n_estimators= 13,random_state= 1)

rfmodel.fit(X_train, y_train)

y_predrf = rfmodel.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_predrf))
kf = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

kf.isnull().sum()

myid1 = kf["id"].copy()

X = kf[["chem_0","chem_1","chem_2","chem_3","chem_4","chem_5","chem_6","chem_7","attribute"]]



y = rfmodel.predict(X)



boo1 = pd.DataFrame(y)

ans = pd.concat([myid1,boo1],axis=1)

ans.columns=["id","class"]

ans = ans.set_index("id")

ans.to_csv("ans_rfc.csv")
#grid_param = {

#    'n_estimators': [5,8,10,12,13,14,15,16,17,18,20],

#    'criterion' : ['entropy', 'gini'],

#    'min_samples_split' : [2,10,20,50],

#    'max_depth': range(1,20,2),

#    'min_samples_leaf': [1,2,5,10,40,20,50],

#    'random_state': [0,1,2]

#    

#}





#grid_search = GridSearchCV(estimator = rfmodel,

#                           param_grid = grid_param,

#                           verbose=1,

#                           cv = 3,

#                           n_jobs = -1)

#grid_search = grid_search.fit(X_train, y_train)
#accuracy = grid_search.best_score_

#accuracy
#grid_search.best_params_