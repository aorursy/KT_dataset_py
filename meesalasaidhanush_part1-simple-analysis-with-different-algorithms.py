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
df=pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.head()
df.shape
df['Country'].unique()
df.dtypes
df['Sex'].replace('M',1,inplace=True)
df['Sex'].replace('F',0,inplace=True)
df['Sex'].astype(int)
df.head()
from wordcloud import WordCloud
cloud=WordCloud(width=1440,height=1080).generate(' '.join(df['Country'].astype(str)))
plt.figure(figsize=(15,10))
plt.imshow(cloud)
df.isnull().any()
df['Category'].unique()
df['Category'].replace('P',1,inplace=True)
df['Category'].replace('C',0,inplace=True)
cor=df.corr()
cor
features=['PassengerId','Sex']
x=df[features]
y=df.Survived
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.countplot(x='Sex',data=df)
sns.barplot(x='Sex',y='Age',data=df)
df.hist(column='Age')
sns.heatmap(cor)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
from sklearn.tree import DecisionTreeClassifier
tree= DecisionTreeClassifier()
tree.fit(x_train,y_train)
pred=tree.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))
import xgboost
xgb=xgboost.XGBClassifier()
xgb.fit(x_train,y_train)
pred1=xgb.predict(x_test)
print(accuracy_score(y_test,pred))
from sklearn import svm
sv=svm.SVC(gamma=0.01,C=1,kernel='linear')
sv.fit(x_train,y_train)
pred2=sv.predict(x_test)
print(accuracy_score(y_test,pred2))
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
pred3=knn.predict(x_test)
print(accuracy_score(y_test,pred3))
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
pred4=lr.predict(x_test)
print(accuracy_score(y_test,pred4))
