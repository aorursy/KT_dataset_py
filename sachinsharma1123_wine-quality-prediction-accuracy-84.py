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
df=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(df.isnull())
sns.heatmap(df.corr())
df['quality'].unique()
sns.countplot(x=df['quality'],data=df)
#here we have 6 types of qualities of wine , wwe have to make them fit into two categories i.e good or bad, we will take 1 if quality is greater than 5 else 0
df['quality']=np.where(df['quality']>5,1,0)
df
y=df['quality']

x=df.drop(['quality'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_s=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_s)

    list_1.append(scores)
plt.scatter(range(1,21),list_1)

plt.xlabel('k values')

plt.ylabel('accuracy scores')

plt.show()
sns.barplot(x=list(range(1,21)),y=list_1)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_3=svm.predict(x_test)

score_3=accuracy_score(y_test,pred_3)
score_3
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()

gbc.fit(x_train,y_train)

pred_4=gbc.predict(x_test)

score_4=accuracy_score(y_test,pred_5)
score_4
#clearly randomforest gives the best accuracy score among all