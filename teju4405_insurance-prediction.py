# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
train.head()
train['Vehicle_Age']=train['Vehicle_Age'].replace({'> 2 Years':2,'1-2 Year':1,'< 1 Year':0})
train['Gender']=train['Gender'].replace({'Male':1,'Female':0})
train['Vehicle_Damage']=train['Vehicle_Damage'].replace({'Yes':1,'No':0})
train.head()

train['Vehicle_Age']=train['Vehicle_Age'].values.astype('float')
train['Gender']=train['Gender'].values.astype('float')
train.info()
Train=train.copy()
Train.head()
Train.corr()
sns.heatmap(Train.corr(),vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(30,220,n=200))
plt.scatter(Train['Age'][:20],Train['Vehicle_Age'][:20])
def Age_groups(x):
    if 20<x<=35:
        return 'Young Group'
    if 35<x<=50:
        return 'Adult Group'
    if 50<x<100:
        return 'Older Group'
Train['Age_Group']=Train['Age'].apply(Age_groups)
Train[['Age','Age_Group']].head()
Train.groupby(['Age_Group','Gender','Driving_License'])['Response'].value_counts(normalize=True)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X=Train[['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Policy_Sales_Channel','Vintage']]
y=Train[['Response']]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
print(X_train.shape,y_train.shape)
scaler=MinMaxScaler()
X_train_trans=scaler.fit_transform(X_train)
X_test_trans=scaler.fit_transform(X_test)
lr=LogisticRegression().fit(X_train_trans,y_train)
predict=lr.predict(X_test_trans)
metrics.accuracy_score(y_test,predict)
sns.distplot(X_test)
sns.distplot(X_train)
test=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
test.head()
test['Vehicle_Age']=test['Vehicle_Age'].replace({'> 2 Years':2,'1-2 Year':1,'< 1 Year':0})
test['Gender']=test['Gender'].replace({'Male':1,'Female':0})
test['Vehicle_Damage']=test['Vehicle_Damage'].replace({'Yes':1,'No':0})
test['Vehicle_Age']=test['Vehicle_Age'].values.astype('float')
test['Gender']=test['Gender'].values.astype('float')
test.head()

Test=test.copy()
X=Test[['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Policy_Sales_Channel','Vintage']]
test_predict=lr.predict(X)
df=pd.DataFrame(test_predict,columns=['Response'],index=Test['id'])
merge=pd.merge(Test,df,how='outer',on='id')
merge

pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv')
