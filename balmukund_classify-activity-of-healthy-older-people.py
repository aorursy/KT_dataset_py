import pandas as pd

import numpy as np

import glob

from sklearn.preprocessing import Normalizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
#find all files in the S1_Dataset. Note:- remove the readme file.

path=r'../input/datasets_healthy_older_people/S1_Dataset'

all_files=glob.glob(path+'/*')

print(all_files)
li=[]

for file in all_files:

    #print(file)

    if file.endswith('.txt'):

        continue

    df=pd.read_csv(file,header=None,index_col=None)

    li.append(df)

df1=pd.concat(li,axis=0,ignore_index=True)
df1.columns=['Time','Acc. Front','Acc. vert','Acc. Lat','id','RSSI','Phase','Freq','Activity Label']

df1.head()
#check for null

df1.isnull().values.any()
cols=len(df1.columns)-1

df11=df1.values

X=df11[:, :8]

Y=df11[:,8]
#Standardize the inputs

normalize=Normalizer()

X=normalize.fit_transform(X)
print(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
# going for Logistic Regression

lr=LogisticRegression()

estimator={'solver':('newton-cg','liblinear','lbfgs','sag')}

gsc=GridSearchCV(lr,estimator)

gsc.fit(X_train,Y_train)

y_gsc_pred=gsc.predict(X_test)

print("accuracy gsc= ",accuracy_score(Y_test,y_gsc_pred))

print(gsc.best_estimator_)
# Going for KNN

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,Y_train)

y_pred_knn=knn.predict(X_test)

print("accuracy KNN= ",accuracy_score(Y_test,y_pred_knn))
# Going for Random Forest

rforest=RandomForestClassifier()

rforest.fit(X_train,Y_train)

y_pred_rforest=rforest.predict(X_test)

print("accuracy Random Forest= ",accuracy_score(Y_test,y_pred_rforest))