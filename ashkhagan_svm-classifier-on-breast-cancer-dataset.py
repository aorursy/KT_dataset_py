# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_cancer=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df_cancer.head()
df_cancer.shape
df_cancer.info()
df_cancer.describe()
df_cancer.columns
plt.figure(figsize=(20,12))

sns.heatmap(df_cancer.corr(),annot=True)
df=df_cancer

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder() 

df['diagnosis'] = lb.fit_transform(df['diagnosis'])

df
df.describe().T
X=df.iloc[:,1:32].values

#X

y=df.iloc[:,0].values

#y
sns.pairplot(df,vars=['radius_mean','texture_mean','area_mean','smoothness_mean'],hue='diagnosis')

plt.ioff()
sns.countplot(df['diagnosis'])

plt.ioff()
sns.scatterplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=df)

plt.ioff()
from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape
X_test.shape
from sklearn.svm import SVC

from sklearn.metrics import classification_report,confusion_matrix

svc_model=SVC()

svc_model.fit(X_train,y_train)
y_predict=svc_model.predict(X_test)

y_test
y_predict
cm=confusion_matrix(y_test,y_predict)

sns.heatmap(cm,annot=True)

plt.ioff()
print(classification_report(y_test,y_predict))
from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler(feature_range=(0,1))

X_train_scaled=sc.fit_transform(X_train)

X_test_scaled=sc.fit_transform(X_test)

sns.scatterplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=df)

plt.ioff()
svc_model.fit(X_train_scaled,y_train)
cm=confusion_matrix(y_test,y_predict)

sns.heatmap(cm,annot=True)

plt.ioff()
print(classification_report(y_test,y_predict))