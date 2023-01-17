import os

for dirname,_,filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname,filename))

        
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

from sklearn.model_selection import train_test_split

df_train=pd.read_csv(r'/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')

df_test=pd.read_csv(r'/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip')

print(df_train.head(5))

print(df_test.head(5))

#remove 0 variance columns

df_train=df_train.loc[:,df_train.apply(pd.Series.nunique)!=1]

df_test=df_test.loc[:,df_test.apply(pd.Series.nunique)!=1]

print(df_train.shape,df_test.shape)
#null values 

df_train.isna().any()

df_test.isna().any()
#categorical data to numeric

#df_train=pd.get_dummies(df_train)
#Divide df_train into test and train data

x_train=df_train.drop(['y'],axis=1)

print(x_train.head(5))

y_train=df_train['y']

#print(y_train.head(5))

x_train.dtypes
from sklearn.preprocessing import LabelEncoder , StandardScaler

le=LabelEncoder()

scaler = StandardScaler()

y_train=le.fit_transform(y_train)

for i in x_train.columns:

    x_train[i]=le.fit_transform(x_train[i])

x_train=scaler.fit_transform(x_train)

#on testing data

#x_test=le.fit_transform(x_test)

#train test split

xtrain,xtest,ytrain,ytest=train_test_split(x_train,y_train,test_size =0.3,random_state=42)
print(xtrain.shape)

print(xtest.shape)
#pip install xgboost
#Dimentionality Reduction

from sklearn.decomposition import PCA

sklearn_pca=PCA(n_components=0.95)

sklearn_pca.fit(xtrain)

xtrain_transformed=sklearn_pca.transform(xtrain)

xtest_transformed=sklearn_pca.transform(xtest)

print(xtrain_transformed.shape,xtest_transformed.shape)
#Xgboost

from sklearn import svm

from xgboost import XGBClassifier

from sklearn.model_selection import KFold , cross_val_score

y_mean=np.mean(y_train)

clf=XGBClassifier(max_depth=3,base_score=y_mean,objective='reg:linear')





# seed=7

# num_trees=15

# kfold=KFold(n_splits=10,random_state=seed)

# model=XGBClassifier(n_estimators=num_trees,random_state=seed)

# results=cross_val_score(model,xtrain_transformed,ytrain,cv=kfold)

# print(results.mean())
clf.fit(xtrain_transformed,ytrain)
y_pred = clf.predict(xtest_transformed)
predictions = [round(value) for value in y_pred]

# evaluate predictions

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(ytest, predictions)

print(mse)

print(y_pred)

print(ytest)

print(ytrain)

print(predictions)