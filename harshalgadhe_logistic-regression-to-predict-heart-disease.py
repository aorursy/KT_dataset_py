# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv')

df.drop(['education'],axis=1,inplace=True)

df.head()
plt.figure(figsize=(12,10))

sns.heatmap(df.corr(),annot=True,cmap=plt.cm.plasma)
df.info()
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)

df.isnull().sum(axis = 0)
sns.pairplot(df)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df.drop(['TenYearCHD'],axis=1),df['TenYearCHD'],test_size=0.2,random_state=4)
from sklearn.linear_model import LogisticRegression

LR=LogisticRegression(max_iter=1650)

LR.fit(X_train,y_train)

yhat=LR.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score(yhat,y_test)
ax=confusion_matrix(yhat,y_test)

sns.heatmap(ax,annot=True,cmap=plt.cm.plasma)
from sklearn.neighbors import KNeighborsClassifier
def KNeigh(X_train,X_test,y_train,y_test):

    score=[]

    

    for i in range(1,10):

        KN=KNeighborsClassifier(n_neighbors=i)

        KN.fit(X_train,y_train)

        yhat=KN.predict(X_test)

        score.append(accuracy_score(yhat,y_test))

    max_score=max(score)

    max_score_index=score.index(max_score)+1

    print(f"Max accuracy of the model is: {max_score} when n_neighbors: {max_score_index}")
KNeigh(X_train,X_test,y_train,y_test)