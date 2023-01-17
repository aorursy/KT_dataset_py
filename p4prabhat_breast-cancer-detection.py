# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing lIbraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 
df= pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

df.head()
df.info()
df= df.dropna(axis=1)
#count of malignant and benignant

df["diagnosis"].value_counts()
sns.countplot(df["diagnosis"], label= "counts")
#encoding categorical data

from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

df.iloc[:,1]= le.fit_transform(df.iloc[:,1].values)
print(df.iloc[:,1])
df.head()
df.corr()
#heatmap

plt.figure(figsize=(20,20))

sns.heatmap(df.corr(),annot= True , fmt = ".0%")
#train test split

from sklearn.model_selection import train_test_split

X= df.drop(["diagnosis"],axis=1)

Y= df.diagnosis.values

X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)
#support vector classifier

from sklearn.svm import SVC

svm= SVC(random_state=1)

svm.fit(X_train, Y_train)

print("SVC accuracy : {:.2f}%".format(svm.score(X_test,Y_test)*100))
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,Y_train)

print("Naive accuracy : {:.2f}%".format(nb.score(X_test,Y_test)*100))
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators=1000,random_state= 1)

rf.fit(X_train,Y_train)

print("Random Forest accuracy : {:.2f}%".format(rf.score(X_test,Y_test)*100))
import xgboost

xg= xgboost.XGBClassifier()

xg.fit(X_train,Y_train)

print("XG Boost accuracy : {:.2f}%".format(xg.score(X_test,Y_test)*100))
#accuracy for the XG Boost is the maximum