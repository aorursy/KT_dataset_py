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
train=pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv')

test=pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv')
train.head()
train.shape
train.info()
train.describe()
train.isnull()
train.isnull().sum().sort_values(ascending=False)
import seaborn as sns
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
train['flag'].value_counts()
sns.countplot(x='flag',data=train)


sns.distplot(train['velocityFront'])




sns.distplot(train['trackingDeviationFront'])


sns.distplot(train['currentBack'])


sns.distplot(train['motorTempBack'])


sns.distplot(train['refPositionBack'])


sns.distplot(train['refVelocityBack'])


sns.distplot(train['positionBack'])
sns.distplot(train['trackingDeviationBack'])
sns.boxplot('positionFront',data=train,palette='rainbow')
sns.boxplot('currentBack',data=train,palette='rainbow')
sns.boxplot('currentFront',data=train,palette='rainbow')
sns.boxplot('refPositionFront',data=train,palette='rainbow')
sns.boxplot('positionBack',data=train,palette='rainbow')
sns.boxplot('refPositionBack',data=train,palette='rainbow')
sns.boxplot('refVelocityFront',data=train,palette='rainbow')
sns.boxplot('refVelocityBack',data=train,palette='rainbow')
sns.boxplot('velocityBack',data=train,palette='rainbow')
sns.boxplot('velocityFront',data=train,palette='rainbow')
import matplotlib.pyplot as plt

%matplotlib inline
train.plot.box()

plt.xticks(list(range(len(train.columns))),train.columns,rotation='vertical')
train1=train.drop(['flag'],axis=1)
for i in train1.columns:

    lw=train1[i].quantile(.10)

    uw=train1[i].quantile(.90)

    train1[i]=np.where(train1[i]<lw,lw,train1[i])

    train1[i]=np.where(train1[i]>uw,uw,train1[i])
for j in train1.columns:

    train1[j]=train1[j].map(lambda i: np.log(i) if i >0 else 0)
for i in test.columns:

    lw=test[i].quantile(.10)

    uw=test[i].quantile(.90)

    test[i]=np.where(test[i]<lw,lw,test[i])

    test[i]=np.where(test[i]>uw,uw,test[i])
for j in test.columns:

    test[j]=test[j].map(lambda i: np.log(i) if i >0 else 0)
corrmat=train.corr()

top_corr_features=corrmat.index

plt.figure(figsize=(15,15))



g=sns.heatmap(train[top_corr_features].corr(),annot=True,cmap='RdYlGn')
new_test=test[['currentBack','trackingDeviationBack','trackingDeviationBack','positionFront','refPositionFront','refPositionBack']]
X=new_train[['currentBack','trackingDeviationBack','trackingDeviationBack','positionFront','refPositionFront','refPositionBack']]

y=train['flag']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=50)

model.fit(X_train,y_train)
model.score(X_train,y_train)
pred=model.predict(X_test)



pred
from sklearn.metrics import accuracy_score,f1_score
accuracy_score(pred,y_test)

f1_score(pred,y_test)
pred=model.predict(new_test)

pred
sample=pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')

sample['flag']=pred

sample.head()
sample.to_csv('Sample Submission final1.csv',index=False)