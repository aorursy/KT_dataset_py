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
# data analysis and wrangling

import pandas as pd

import numpy as np

import random



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]

train_df
train_df = train_df.drop(['PassengerId','Cabin','Ticket'], axis=1)

test_df = test_df.drop(['PassengerId','Cabin','Ticket'], axis=1)



#complete missing age with median

train_df['Age'].fillna(train_df['Age'].median(), inplace = True)

test_df['Age'].fillna(test_df['Age'].median(), inplace = True)



#complete embarked with mode

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)



test_df['Fare'].fillna(test_df['Fare'].mode()[0], inplace = True)



train_df
train_df['Sex'] = train_df['Sex'].map({'male':0,'female':1})

test_df['Sex'] = test_df['Sex'].map({'male':0,'female':1})



train_df
for dataset in [train_df,test_df]:

    

    # Creating a categorical variable to tell if the passenger is alone

    dataset['IsAlone'] = ''

    dataset['IsAlone'].loc[((dataset['SibSp'] + dataset['Parch']) > 0)] = 1

    dataset['IsAlone'].loc[((dataset['SibSp'] + dataset['Parch']) == 0)] = 0

    

    

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # take only top 10 titles

    title_names = (dataset['Title'].value_counts() < 10) #this will create a true false series with title name as index



    #apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    print(dataset['Title'].value_counts())

    

    dataset.drop(['Name','SibSp','Parch'],axis=1,inplace=True)



train_df.head()
#define x and y variables for dummy features original

train_dummy = pd.get_dummies(train_df,drop_first=True)

test_dummy = pd.get_dummies(test_df,drop_first=True)



train_dummy
X_final = train_dummy.drop(['Survived'],axis=1).values # for original features

target = train_dummy['Survived'].values

X_final
target.shape
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_final = sc.fit_transform(X_final)

print(X_final.shape,'\n',X_final)
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs



class MYSVM:

    def __init__(self,lr=.005,iters=1000,lambda_p=0.01):

        self.lr=lr

        self.iters=iters

        self.lambda_p = lambda_p



        self.w = None

        self.b = None

        

    def fit(self,X,y):

        n_samples,n_feat = X.shape

        

        self.w = np.zeros(n_feat)

        self.b = 0

        

        y_true = np.where(y <= 0, -1, 1) # same as -1 if y <= 0 else 1

        

        for _ in range(self.iters):

            for idx, sample in enumerate(X):

                # do a forward pass

                y_pred = y_true[idx]*(np.dot(sample,self.w)-self.b)

                

                # update with the graient

                if y_pred>=1:

                    self.w -= self.lr * (2 * self.lambda_p * self.w)

                else:

                    self.w -= self.lr * (2 * self.lambda_p * self.w - np.dot(sample, y_true[idx]))

                    self.b -= self.lr * y_true[idx]

        

    def predict(self,X):

        # do a forward pass

        y = np.dot(X, self.w) - self.b

        return np.sign(y) # if -1, belongs to class 0, else class 1

        

# X,y = make_blobs(centers = 3,n_samples = 1000, n_features = 3, shuffle = False)

svm_clf = MYSVM()
svm_clf.fit(X_final,target)
preds = svm_clf.predict(X_final) 

preds[:10]
preds_ = np.where(preds==-1,0,1)

accurate = 0

for i,val in enumerate(target):

    if val==preds_[i]:

        accurate+=1

print(accurate)

print("Accuracy = ",accurate/len(preds))
from sklearn.svm import SVC

svm_sk = SVC(gamma='auto').fit(X_final,target)

svm_preds = svm_sk.predict(X_final)

svm_preds[:10]
accurate = 0

for i,val in enumerate(target):

    if val==svm_preds[i]:

        accurate+=1

print(accurate)

print("Accuracy = ",accurate/len(svm_preds))