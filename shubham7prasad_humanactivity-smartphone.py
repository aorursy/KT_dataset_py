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
train=pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')

test=pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/test.csv')



data=train

data.head()
missingValues=data.isnull().sum()

print(missingValues[missingValues>0])
#Check Duplicates

print('Number of duplicates in train:',sum(train.duplicated()))

print('Number of duplicates in test:',sum(test.duplicated()))
data.info()
data.describe()
#Check for unique values

data.nunique()
data.Activity.unique()
#Check for Class Imbalance

import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(12,7))

sns.countplot('Activity',data=data)
#plt.figure(figsize=(12,8))

g=sns.FacetGrid(train,hue='Activity',height=5,aspect=3)

g.map(sns.distplot,'tBodyAccMag-mean()').add_legend()
sns.distplot(a=data.subject,kde=False)
sns.scatterplot(x='subject',y='tBodyAccMag-mean()',hue='Activity',data=train)
plt.figure(figsize=(12,7))

sns.violinplot('Activity','angle(tBodyAccMean,gravity)',data=train,split=True)
sns.swarmplot('Activity','angle(tBodyAccMean,gravity)',data=train)
if(tBodyAccMag-mean()<=-0.5):

    Activity = "static"

else:

    Activity = "dynamic"
plt.figure(figsize=(12,8))

plt.subplot(1,2,1)



sns.distplot(train[train['Activity']=='SITTING']['tBodyAccMag-mean()'])

sns.distplot(train[train['Activity']=='STANDING']['tBodyAccMag-mean()'])

sns.distplot(train[train['Activity']=='LAYING']['tBodyAccMag-mean()'])



plt.subplot(1,2,2)

plt.title("Dynamic Activities(closer view)")

sns.distplot(train[train["Activity"]=="WALKING"]['tBodyAccMag-mean()'],hist = False, label = 'Walking')

sns.distplot(train[train["Activity"]=="WALKING_DOWNSTAIRS"]['tBodyAccMag-mean()'],hist = False,label = 'Downstairs')

sns.distplot(train[train["Activity"]=="WALKING_UPSTAIRS"]['tBodyAccMag-mean()'],hist = False, label = 'Upstairs')


plt.figure(figsize=(10,7))

sns.boxplot(x='Activity',y='tBodyAccMag-mean()',data=train)
if(tBodyAccMag-mean()<=-0.8):

    Activity = "static"

if(tBodyAccMag-mean()>=-0.6):

    Activity = "dynamic"

from sklearn.manifold import TSNE



X_tsne=train.drop(['Activity','subject'],axis=1)



tsne=TSNE(random_state=42,n_components=2,verbose=1,perplexity=50,n_iter=1000).fit_transform(X_tsne)
train.head()
plt.figure(figsize=(12,8))

sns.scatterplot(x=tsne[:,0],y=tsne[:,1],hue=train['Activity'])
X_train = train.drop(['subject', 'Activity'], axis=1)

y_train = train.Activity

X_test = test.drop(['subject', 'Activity'], axis=1)

y_test = test.Activity

print('Training data size : ', X_train.shape)

print('Test data size : ', X_test.shape)
from sklearn. linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



model=LogisticRegression()



model.fit(X_train,y_train)

yhat=model.predict(X_test)



lr_accuracy = accuracy_score(y_true=y_test, y_pred=yhat)

print("Accuracy using Logistic Regression : ", lr_accuracy)