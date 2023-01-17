import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df=pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")
df.head()
df['gender'].value_counts()
df.isnull().sum()
df.dtypes
df.describe()
df['Satisfied'].value_counts()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder 

  

le = LabelEncoder() 



for col in df.select_dtypes(include='object').columns:

    if(col!="TotalCharges"):

        df[col]=le.fit_transform(df[col]) 
df.dtypes
data=df.drop(['custId','Satisfied','TotalCharges'],axis=1)
from sklearn.preprocessing import MinMaxScaler



scaler=MinMaxScaler()

X_std = scaler.fit_transform(data)
from sklearn.cluster import KMeans

model = KMeans(n_clusters=2,random_state=21)

model.fit(X_std)
test=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
test.shape
test.isnull().sum()
test.dtypes
for col in test.select_dtypes(include='object').columns:

    if(col!="TotalCharges"):

        test[col]=le.fit_transform(test[col]) 
data1=test.drop(['custId','TotalCharges'],axis=1)
test_std = scaler.transform(data1)
val=model.predict(test_std)
val
compare = pd.DataFrame({'custId': test['custId'], 'Satisfied' : val})
compare.shape
compare.to_csv('submission2.csv',index=False)
val1=model.fit_predict(test_std)
compare = pd.DataFrame({'custId': test['custId'], 'Satisfied' : val1})
val1
compare.to_csv('submission3.csv',index=False)