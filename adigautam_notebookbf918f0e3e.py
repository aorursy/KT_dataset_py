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
data_train=pd.read_csv('/kaggle/input/car-crash-dataset/train-new.csv')
data_test=pd.read_csv('/kaggle/input/car-crash-dataset/test-new.csv')
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
data_train['dvcat'].value_counts()

sns.countplot(data_train['dvcat'])
sum(data_train['dvcat'].isnull())
data_train['dead'].value_counts()
sns.countplot(data_train['dead'])
data_train['airbag'].describe()
data_train['sex'].describe()
data_train['yearacc'].value_counts()
data_train['yearVeh'].value_counts()
data_train['abcat'].value_counts()
data_train['occRole'].value_counts()
data_train['deploy'].value_counts()
sns.pairplot(data_train)
data_train.isnull().sum()

data_train['seatbelt'].value_counts()
sns.barplot(data=data_train,x='dead',y='injSeverity',estimator=np.std)
sns.barplot(data=data_train,x='dead',y='injSeverity',)
sns.boxplot(x='injSeverity',y='dead',data=data_train,hue='seatbelt')
sns.boxplot(x='injSeverity',y='seatbelt',data=data_train)
sns.heatmap(data_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='dead',hue='injSeverity',data=data_train,)
sex = pd.get_dummies(data_train['sex'],drop_first=True)
dead=pd.get_dummies(data_train['dead'],drop_first=True)
airbag=pd.get_dummies(data_train['airbag'],drop_first=True)
seatbelt=pd.get_dummies(data_train['seatbelt'],drop_first=True)
abcat=pd.get_dummies(data_train['abcat'],drop_first=True)
occRole=pd.get_dummies(data_train['occRole'],drop_first=True)
dvcat=pd.get_dummies(data_train['dvcat'],drop_first=True)
def replace(x):
    if x=='none':
        return 'Noairbag'
    else:
        return x
def seatreplace(x):
    if x=='none':
        return 'notbelted'
    else:
        return x
data_train['airbag']=data_train['airbag'].apply(replace)
data_train['seatbelt']=data_train['seatbelt'].apply(seatreplace)
data_train.head()
data_train.drop(['sex','dead','airbag','seatbelt','abcat','occRole','dvcat'],axis=1,inplace=True)
data_train=pd.concat([data_train,sex,dead,airbag,seatbelt,abcat,occRole,dvcat],axis=1)
data_train.info()
data_train.head()
data_train.drop('caseid',axis=1,inplace=True)
data_train.head()
data_train['ageVeh']=data_train['yearacc']-data_train['yearVeh']
data_train.head()
data_train.drop(['yearacc','yearVeh'],axis=1,inplace=True)
data_train.head()
import statsmodels.api as sm
y=data_train['injSeverity']
x=data_train.drop(['injSeverity'],axis=1)
xc=sm.add_constant(x)
mlogit=sm.MNLogit(y,xc)
fmlogit=mlogit.fit()
print(fmlogit.summary())
data_test
sex = pd.get_dummies(data_test['sex'],drop_first=True)
dead=pd.get_dummies(data_test['dead'],drop_first=True)
airbag=pd.get_dummies(data_test['airbag'],drop_first=True)
seatbelt=pd.get_dummies(data_test['seatbelt'],drop_first=True)
abcat=pd.get_dummies(data_test['abcat'],drop_first=True)
occRole=pd.get_dummies(data_test['occRole'],drop_first=True)
dvcat=pd.get_dummies(data_test['dvcat'],drop_first=True)
data_test['airbag']=data_test['airbag'].apply(replace)
data_test['seatbelt']=data_test['seatbelt'].apply(seatreplace)
data_test.drop(['sex','dead','airbag','seatbelt','abcat','occRole','dvcat'],axis=1,inplace=True)
data_test=pd.concat([data_test,sex,dead,airbag,seatbelt,abcat,occRole,dvcat],axis=1)
data_train.head()
data_test.head()
data_test.drop('caseid',axis=1,inplace=True)
data_test['ageVeh']=data_test['yearacc']-data_test['yearVeh']
data_test.drop(['yearacc','yearVeh'],axis=1,inplace=True)


