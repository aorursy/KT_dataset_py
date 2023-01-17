import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

%matplotlib inline
data = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

print(data.head(3))

data.shape
dt_ctg=data.groupby('Category',as_index=False)['Rating'].mean()

dt_ctg.head()
dt_rvws=data.groupby('Reviews',as_index=False)['Rating'].mean()

dt_rvws.head(10)
dt_sz=data.groupby('Size',as_index=False)['Rating'].mean()

dt_sz.head(10)
dt_in=data.groupby('Installs',as_index=False)['Rating'].mean()

dt_in.head(6)
dt_gn=data.groupby('Genres',as_index=False)['Rating'].mean()

dt_gn.head(10)
dt_tp=data.groupby('Type',as_index=False)['Rating'].mean()

dt_tp.head()
dt_prc=data.groupby('Price',as_index=False)['Rating'].mean()

dt_prc.head(10)
dt_cr=data.groupby('Content Rating',as_index=False)['Rating'].mean()

dt_cr.head(10)
dt_av=data.groupby('Android Ver',as_index=False)['Rating'].mean()

dt_av.head()
sns.countplot(x='Type',data=data)
plt.figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')



sns.barplot(x='Content Rating', y='Rating', hue="Type", data=data, estimator=np.median)

plt.show()
plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')



sns.boxplot(x='Content Rating', y='Rating', hue="Type", data=data)

plt.show()
plt.figure(figsize=(16,8))

sns.countplot(y='Category',data=data)

plt.show()
plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k')



sns.barplot(y='Category', x='Rating', hue="Type", data=data, estimator=np.median)

plt.show()
sns.countplot(x='Content Rating',data=data)
sns.boxplot(x='Content Rating', y='Rating', data=data)
plt.figure(figsize=(7,14))

sns.barplot(y='Installs', x='Rating', data=data)

plt.show()
plt.figure(figsize=(10, 25))

sns.barplot(y='Android Ver', x='Rating', data=data)

plt.show()
plt.figure(figsize=(8, 15))

sns.countplot(y='Rating',data=data )

plt.show()
data[data['Rating'] == 19]
data[10470:10475]
data.iloc[10472,1:] = data.iloc[10472,1:].shift(1)

data[10470:10475]
data.isnull().sum()
total=data.isnull().sum()

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(13)
data.dropna(inplace=True)
data.shape
data.head(3)
catgry=pd.get_dummies(data['Category'],prefix='catg',drop_first=True)

typ=pd.get_dummies(data['Type'],prefix='typ',drop_first=True)

cr=pd.get_dummies(data['Content Rating'],prefix='cr',drop_first=True)

frames=[data,catgry,typ,cr]

data=pd.concat(frames,axis=1)

data.drop(['Category','Installs','Type','Content Rating'],axis=1,inplace=True)
data.drop(['App','Size','Price','Genres','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)
data.head(3)
X=data.drop('Rating',axis=1)

y=data['Rating'].values

y=y.astype('int')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=625)
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.transform(X_test)
'''ts_score=[]

import numpy as np

for j in range(1000):# random state depend upon size of the data set

    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=j,test_size=.1)

    lr=LogisticRegression()

    i= lr.fit(X_train,y_train)

    ts_score.append(i.score(X_test,y_test))

    

k=ts_score.index(np.max(ts_score))'''
lr_c=LogisticRegression(random_state=0)

lr_c.fit(X_train,y_train)

lr_pred=lr_c.predict(X_test)

lr_cm=confusion_matrix(y_test,lr_pred)

lr_ac=accuracy_score(y_test, lr_pred)

print('LogisticRegression_accuracy:',lr_ac)
plt.figure(figsize=(10,5))

plt.title("lr_cm")

sns.heatmap(lr_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.show()
y_pred=lr_c.predict(X_test)
from sklearn.metrics import confusion_matrix

con_mat=confusion_matrix(y_test,y_pred)

print(con_mat)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
import seaborn as sb

cm=pd.DataFrame(con_mat)

sb.heatmap(cm,annot=True,fmt='d')