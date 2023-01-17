# -*- coding: utf-8 -*-

"""

Created on Thu Mar  7 12:19:25 2019



@author: SIDDIK

"""

#1.Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#2.Data Preprocessing

#2.1.Data Import

df=pd.read_csv("../input/veriler.csv")

boy=df[["boy"]]

boykilo=df[["boy","kilo"]]



#2.2.Dealing with missing values

df=pd.read_csv("../input/eksikveriler.csv")

from sklearn.impute import SimpleImputer

imputer=SimpleImputer()

yas=df.iloc[:,1:4].values

imputer=imputer.fit(yas)

yas=imputer.transform(yas)



#2.3.Data Encode Categorical -> Numerical

ulke=df.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

ulke[:,0]=le.fit_transform(ulke[:,0])



from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder()

ulke=ohe.fit_transform(ulke).toarray()



#2.4.Transform Numpy arrays to Dataframe

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])

sonuc2=pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])



cinsiyet=df.iloc[:,-1:].values

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])



#2.5.Data Concatenate

s=pd.concat([sonuc,sonuc2],axis=1)

s2=pd.concat([s,sonuc3],axis=1)



#2.6.Data Split for training and testing 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)



#2.7.Data Scale(Standardization)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)
