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
eval1=pd.read_csv('/kaggle/input/the-insurance-company-tic-benchmark/tic_2000_eval_data.csv')

eval1.head()
col1=eval1.columns.values

col1
train=pd.read_csv('/kaggle/input/the-insurance-company-tic-benchmark/tic_2000_train_data.csv').dropna()

train
col2=train.columns.values

col2
from matplotlib import pyplot as plt

import seaborn as sns
train.hist(column='MOSTYPE',bins=10,color='violet',grid=False)

plt.box(on=None)
sns.scatterplot(data=train,x='MOPLMIDD',y='MGEMLEEF')
sns.distplot(train['MRELGE'])
sns.barplot(data=train,x='MOSHOOFD',y='MOSTYPE',hue='CARAVAN')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

lr=LogisticRegression(C=0.01,solver='sag')

y=train[['CARAVAN']]

X=train.iloc[:,:85]

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4,test_size=0.2)
print(X_train.shape)

print(X_test.shape)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)

pred
from sklearn import metrics

print('Accuracy train set' ,metrics.accuracy_score(y_train,lr.predict(X_train)))

print('Accuracy test set',metrics.accuracy_score(y_test,pred))
eval2=eval1.iloc[:,:85]

result=lr.predict(eval2)

df=pd.DataFrame(result)

df

from sklearn.neighbors import KNeighborsClassifier

neigh=KNeighborsClassifier(n_neighbors=3,algorithm='auto').fit(X_train,y_train)

neigh.predict(X_test)
metrics.accuracy_score(y_test,neigh.predict(X_test))
df1=neigh.predict(eval2)

pd.DataFrame(df1)
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion='entropy',max_depth=4).fit(X_train,y_train)

dtc.predict(X_test)
dtc.predict(eval2)
print(metrics.accuracy_score(y_test,dtc.predict(X_test)))
from sklearn.cluster import KMeans 

from sklearn.datasets.samples_generator import make_blobs 
clusternum=3

means=KMeans(init='k-means++',n_clusters=clusternum,n_init=12)

means.fit(X)
labels=means.labels_

print(labels)
train['CLUSTERS']=labels

train
train.groupby('CLUSTERS').mean()
area=np.pi*(X[:,1])**2

plt.scatter(X[:,0],X[:,3],s=area,c=labels.astype(float),alpha=1.5)

plt.xlabel('features')

plt.ylabel('Caravan',fontsize=10)

plt.show()