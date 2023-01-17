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
import pandas as pd
df = pd.read_csv("../input/IRIS.csv")
df.head()
from sklearn.model_selection import train_test_split
x = df.iloc[ :, :-1]

y = df.iloc[:, 4]

print(x.shape)

print (y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print(x_train.shape)

print(y_train.shape)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
knn.score(x_test,y_test)
import matplotlib.pyplot as plt
plt.figure()
import seaborn as sns
sns.pairplot(data= df,hue='species',palette='RdBu')
plt.xticks([0,1],['features','species'])
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
print(logreg)