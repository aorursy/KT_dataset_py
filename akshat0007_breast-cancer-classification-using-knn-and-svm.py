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
df=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.head(7)
df.dropna()
df.isnull().sum()
df.describe()
df.corr()
x=df[["radius_mean","texture_mean","smoothness_mean","compactness_mean","concavity_mean"]]
y=df["diagnosis"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()



logmodel.fit(x_train,y_train)

predictions=logmodel.predict(x_test)
df1=x_test
len(x_test)
df1
predictions
from sklearn.metrics import jaccard_similarity_score
accuracy_score=jaccard_similarity_score(y_test,predictions)

print(accuracy_score*100)
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,predictions)

print(matrix)
from sklearn import svm
clf=svm.SVC(gamma="scale")
clf.fit(x_train,y_train)
predictions=clf.predict(x_test)
accuracy_score=jaccard_similarity_score(y_test,predictions)

print(accuracy_score*100)