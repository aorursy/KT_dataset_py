# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)

#for visualization

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
data=pd.read_csv('../input/placement-visualization-prediction/datasets_placememt.csv')
data
data.shape
data.head()
data.tail()
data.info()
#Check the datatypes of this dataset

data.dtypes
#Check the null values

data.isnull().sum()
data.describe(include='all')
#clean the nan value

data['salary'].fillna(data['salary'].mean(), inplace=True)
#after cleaning the nan value check the null value

data.isnull().sum()
sns.countplot(data.status)

plt.show()
data['status'].value_counts()
sns.countplot(data.gender,hue=data.status)

plt.show()
sns.distplot(data.ssc_p[data.status=='Not Placed'])

sns.distplot(data.ssc_p[data.status=='Placed'])

plt.legend(['Not placed','placed'])

plt.show()
data['ssc_b'].value_counts()
sns.countplot(data.ssc_b,hue=data.status)

plt.show()
sns.distplot(data.hsc_p[data.status=='Not Placed'])

sns.distplot(data.hsc_p[data.status=='Placed'])

plt.legend(['Not placed','placed'])

plt.show()
data['hsc_b'].value_counts()
sns.countplot(data.hsc_b,hue=data.status)

plt.show()
data['hsc_s'].value_counts()
sns.countplot(data.hsc_s,hue=data.status)

plt.show()
sns.distplot(data.degree_p[data.status=='Not Placed'])

sns.distplot(data.degree_p[data.status=='Placed'])

plt.legend(['Not placed','placed'])

plt.show()
data['degree_t'].value_counts()
sns.countplot(data.degree_t,hue=data.status)

plt.show()
data['workex'].value_counts()
sns.countplot(data.workex,hue=data.status)

plt.show()
sns.distplot(data.etest_p[data.status=='Not Placed'])

sns.distplot(data.etest_p[data.status=='Placed'])

plt.legend(['Not placed','placed'])

plt.show()
data['specialisation'].value_counts()
sns.countplot(data.specialisation,hue=data.status)

plt.show()
sns.distplot(data.mba_p[data.status=='Not Placed'])

sns.distplot(data.mba_p[data.status=='Placed'])

plt.legend(['Not placed','placed'])

plt.show()
sns.swarmplot(x=data.salary,y=data.gender,hue=data.status,data=data)

plt.show()
#drop the unrelevent column

data.drop(['sl_no'],axis=1,inplace=True)
#Label encoder

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

data.gender=lb.fit_transform(data.gender)



le1=LabelEncoder()

data.ssc_b=le1.fit_transform(data.ssc_b)



lb2=LabelEncoder()

data.hsc_b=lb2.fit_transform(data.hsc_b)



le3=LabelEncoder()

data.hsc_s=le3.fit_transform(data.hsc_s)



lb4=LabelEncoder()

data.degree_t=lb4.fit_transform(data.degree_t)



le5=LabelEncoder()

data.workex=le5.fit_transform(data.workex)



lb6=LabelEncoder()

data.specialisation=lb6.fit_transform(data.specialisation)



le7=LabelEncoder()

data.status=le7.fit_transform(data.status)
data
corr=data.corr()

plt.figure(figsize=(12,5))

sns.heatmap(corr,annot=True,cmap='coolwarm')

plt.show()
#input and output selection

ip=data.drop(['status'],axis=1)

op=data['status']
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



ct = ColumnTransformer([('status',OneHotEncoder(),[5,7])],remainder='passthrough')



ip = np.array(ct.fit_transform(ip),dtype = np.str)
from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(ip,op,test_size=0.1)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(xtr)

xtr=sc.transform(xtr)

xts=sc.transform(xts)
from sklearn.linear_model import LogisticRegression

alg=LogisticRegression()
#train the algorithm with the training data

alg.fit(xtr,ytr)

yp=alg.predict(xts)
from sklearn import metrics

cm=metrics.confusion_matrix(yts,yp)

print(cm)
accuracy=metrics.accuracy_score(yts,yp)

print(accuracy)
precission=metrics.precision_score(yts,yp)

print(precission)
recall=metrics.recall_score(yts,yp)

print(recall)