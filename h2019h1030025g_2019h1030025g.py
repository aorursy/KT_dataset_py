import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from sklearn import datasets, linear_model
df=pd.read_csv('../input/eval-lab-1-f464-v2/train.csv')
df.describe()
df.isnull().sum()
df.info()
df.replace(" ", np.nan, inplace = True)

df["feature3"]=df["feature3"].fillna(df["feature3"].mean())

df["feature4"]=df["feature4"].fillna(df["feature4"].mean())

df["feature5"]=df["feature5"].fillna(df["feature5"].mean())

df["feature8"]=df["feature8"].fillna(df["feature8"].mean())

df["feature9"]=df["feature9"].fillna(df["feature9"].mean())

df["feature10"]=df["feature10"].fillna(df["feature10"].mean())

df["feature11"]=df["feature11"].fillna(df["feature11"].mean())

df["type"]=np.where(df["type"].str.contains("new"),1,0)
pf=pd.read_csv("../input/eval-lab-1-f464-v2/test.csv")



pf.replace("", np.nan, inplace = True)

pf["feature3"]=pf["feature3"].fillna(pf["feature3"].mean())

pf["feature4"]=pf["feature4"].fillna(pf["feature4"].mean())

pf["feature5"]=pf["feature5"].fillna(pf["feature5"].mean())

pf["feature8"]=pf["feature8"].fillna(pf["feature8"].mean())

pf["feature9"]=pf["feature9"].fillna(pf["feature9"].mean())

pf["feature10"]=pf["feature10"].fillna(pf["feature10"].mean())

pf["feature11"]=pf["feature11"].fillna(pf["feature11"].mean())

pf["type"]=np.where(pf["type"].str.contains("new"),1,0)
df.info()

pf.info()
features=["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","type","feature10","feature11"]

x=df[features]

y=df["rating"]

zz=pf[features]
###from sklearn.model_selection import GridSearchCV

#from sklearn.neighbors import KNeighborsClassifier

#making the instance

#model = KNeighborsClassifier(n_jobs=-1)

#Hyper Parameters Set

#params = {'n_neighbors':[5,6,7,8,9,10],

 #         'leaf_size':[1,2,3,5],

 #         'weights':['uniform', 'distance'],

 #         'algorithm':['auto', 'ball_tree','kd_tree','brute'],

 #         'n_jobs':[-1]}

#Making models with hyper parameters sets

#model1 = GridSearchCV(model, param_grid=params, n_jobs=1)

#Learning

#model1.fit(train_X,train_y)

#The best hyper parameters set

#print("Best Hyper Parameters:\n",model1.best_params_)

#Prediction

#prediction=model1.predict(test_X)

#importing the metrics module

#from sklearn import metrics

#evaluation(Accuracy)

#print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

#print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))
#from sklearn.model_selection import GridSearchCV

#parameters = [{'C': [1, 10, 1000], 'kernel': ['linear']},

 #             {'C': [1, 10, 1000], 'kernel': ['rbf'], 'gamma': [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1]}]

#grid_search = GridSearchCV(estimator = classifier,

 #                          param_grid = parameters,

  #                         scoring = 'accuracy',

   #                        cv = 10,

    #                       n_jobs = -1)

#grid_search = grid_search.fit(X_train, y_train)

from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y = train_test_split(x,y,test_size=0.33,random_state=6) 

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

train_X[features] = scaler.fit_transform(train_X[features])

test_X[features] = scaler.transform(test_X[features])

zz[features] = scaler.transform(zz[features])

from sklearn.neighbors import KNeighborsClassifier

model1 = KNeighborsClassifier(n_neighbors=9,leaf_size=6,algorithm='auto',n_jobs=-1,p=30,weights='distance',metric='euclidean')

from sklearn.svm import SVC

model2 =SVC(C=10,kernel='rbf',gamma=1.84,random_state=8,degree=10,probability=True,decision_function_shape='ovr')

from sklearn.tree import DecisionTreeClassifier

model3 =DecisionTreeClassifier(random_state=0,criterion="entropy")

from sklearn.linear_model import LogisticRegression

model4=LogisticRegression(C=2.0,random_state=0,solver='sag',multi_class='multinomial')

from sklearn.ensemble import VotingClassifier

model = VotingClassifier(estimators=[('kn', model1), ('svc', model2),('dt', model3),('lr', model4)], voting='hard')
model.fit(train_X,train_y)

score=model.score(test_X,test_y)

score

yyy=model.predict(zz)

yyy

np.savetxt("en_final", yyy, delimiter=",")

