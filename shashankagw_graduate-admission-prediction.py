# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv(r"../input/Admission_Predict.csv")
column_names = df.columns

column_names
df.info()
df.head()
df.drop('Serial No.',axis=1,inplace=True)

df.head()
df.rename(columns = {'Chance of Admit ':'Chance of Admit'},inplace=True)
import matplotlib.pyplot as plt

import seaborn as sns
fig=plt.figure(figsize=(15,5))

ax1 = plt.subplot(121)

ax2 = plt.subplot(122)

sns.countplot(x='University Rating',data = df,ax=ax1)

sns.countplot(x='Research',data = df,ax=ax2)

plt.title("Research Experience")
df['GRE_Score_bin'] = pd.cut(df['GRE Score'],5) 

df['TOEFL_Score_bin'] = pd.cut(df['TOEFL Score'],5)

df.head()
fig=plt.figure(figsize=(15,5))

ax1 = plt.subplot(121)

ax2 = plt.subplot(122)

sns.countplot(x='GRE_Score_bin',data= df,ax=ax1)

sns.countplot(x='TOEFL_Score_bin',data= df,ax=ax2)
fig=plt.figure(figsize=(15,15))

ax1 = plt.subplot(111)

sns.heatmap(df.corr(),ax=ax1,annot=True,linewidth=0.05, fmt= '.2f',cmap="magma")
df.head()
df1 = df.drop(['GRE_Score_bin','TOEFL_Score_bin'],axis=1)

df1.head()
df2 = df.drop(['GRE Score','TOEFL Score'],axis=1)

df2.head()
X = df1.drop(['Chance of Admit'],axis=1)

y=df1['Chance of Admit']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

lrg = LinearRegression()

lrg.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error

y_predict = lrg.predict(X_test)
print("Accuracy Level of training set : ",lrg.score(X_train, y_train)*100,' %')

print("Accuracy Level of test set : ",lrg.score(X_test, y_test)*100,' %')

print("Mean Squared Error is : ",np.sqrt(mean_squared_error(y_test,y_predict)))