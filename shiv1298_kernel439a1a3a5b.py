# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
df=pd.read_csv('/kaggle/input/performance-prediction/summary.csv')
df
df.describe()
import seaborn as sns

sns.heatmap(df.isnull())
df.isnull().sum()
df['3PointPercent']=df['3PointPercent'].fillna(df['3PointPercent'].mean())
df.isnull().sum()
df.corr()['Target']
df=df.drop(['Name'],axis=1)
y=df['Target']

x=df.drop(['Target'],axis=1)
x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

list_1=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_y=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_y)

    list_1.append(scores)
import matplotlib.pyplot as plt

plt.plot(range(1,11),list_1)

plt.show()
print(max(list_1))
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score1=accuracy_score(y_test,pred_1)
print(score1)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,pred_1))