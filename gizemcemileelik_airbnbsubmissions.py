# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
train=pd.read_csv('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')

test = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/test_users.csv.zip')
submissionid=test.id


# For identification purposes

train.loc[:,'Train'] = 1

test.loc[:,'Train'] = 0



test['country_destination'] = 0



df = pd.concat([train,test], ignore_index=True)
train=df[df['Train']==1]
test=df[df['Train']==0]
train['age'].fillna(train['age'].mean(), inplace=True)
test['age'].fillna(train['age'].mean(), inplace=True)
train.isnull().sum()
test.isnull().sum()
train['first_affiliate_tracked'] = train['first_affiliate_tracked'].fillna('Unknown')

test['first_affiliate_tracked'] = test['first_affiliate_tracked'].fillna('Unknown')
train.drop(['date_account_created','timestamp_first_active','date_first_booking','Train'],axis=1,inplace=True)
test.drop(['date_account_created','timestamp_first_active','date_first_booking','country_destination','Train'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

  

train['gender']= le.fit_transform(train['gender'])

train['signup_method']= le.fit_transform(train['signup_method']) 

train['first_affiliate_tracked']= le.fit_transform(train['first_affiliate_tracked']) 

train['signup_method']= le.fit_transform(train['signup_method']) 

train['language']= le.fit_transform(train['language'])

train['affiliate_channel']= le.fit_transform(train['affiliate_channel'])

train['affiliate_provider']= le.fit_transform(train['affiliate_provider'])

train['signup_app']= le.fit_transform(train['signup_app'])

train['first_device_type']= le.fit_transform(train['first_device_type'])

train['first_browser']= le.fit_transform(train['first_browser'])

le = LabelEncoder() 

  

test['gender']= le.fit_transform(test['gender'])

test['signup_method']= le.fit_transform(test['signup_method']) 

test['first_affiliate_tracked']= le.fit_transform(test['first_affiliate_tracked']) 

test['signup_method']= le.fit_transform(test['signup_method']) 

test['language']= le.fit_transform(test['language'])

test['affiliate_channel']= le.fit_transform(test['affiliate_channel'])

test['affiliate_provider']= le.fit_transform(test['affiliate_provider'])

test['signup_app']= le.fit_transform(test['signup_app'])

test['first_device_type']= le.fit_transform(test['first_device_type'])

test['first_browser']= le.fit_transform(test['first_browser'])

train.country_destination.replace('NDF',0,inplace=True)

train.country_destination.replace('US',1,inplace=True)

train.country_destination.replace('other',2,inplace=True)

train.country_destination.replace('FR',3,inplace=True)

train.country_destination.replace('CA',4,inplace=True)

train.country_destination.replace('GB',5,inplace=True)

train.country_destination.replace('ES',6,inplace=True)

train.country_destination.replace('IT',7,inplace=True)

train.country_destination.replace('PT',8,inplace=True)

train.country_destination.replace('NL',9,inplace=True)

train.country_destination.replace('DE',10,inplace=True)

train.country_destination.replace('AU',11,inplace=True)
test
from sklearn.model_selection import train_test_split

y=train['country_destination']

X=train.drop(['country_destination','id'],axis=1)

# split the dataset into train and test sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1,shuffle=True,stratify=y )
pred_country={0:"NDF", 1:"US", 2:"other", 3:"FR", 4:"CA", 5:"GB", 6:"ES", 7:"IT", 8:"PT", 9:"DE", 10:"NL", 11:"AU"}
#predictionsrf=rf.predict(test.drop(['id'],axis=1))
#results=[]

#for i in predictionsrf:

#    results.append(pred_country[i])

#print(results)
#my_submissionrf = pd.DataFrame({'id': test.id, 'country':results})

#my_submissionrf.to_csv('submission.csv', index=False)
from sklearn.model_selection import GridSearchCV

parameters = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}

mlpgridsearch = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)

mlpgridsearch.fit(X_train,y_train)

predsgridmlp = mlpgridsearch.predict(X_test)



print(mlpgridsearch.score(X_train, y_train))

print(mlpgridsearch.best_params_)
from sklearn.neural_network import MLPClassifier

#Generate prediction using Neural Net



#mlp = MLPClassifier()

#mlp.fit(X_train,y_train)

#predsmlp = mlp.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print(classification_report(y_test, predsgridmlp, target_names=target_names))
predictionsmlp=mlpgridsearch.predict(test.drop(['id'],axis=1))
results=[]

for i in predictionmlp:

    results.append(pred_country[i])

print(results)
my_submissionmlp = pd.DataFrame({'id': test.id, 'country':results})

my_submissionmlp.to_csv('submission.csv', index=False)
#predictionsxgb=xgb.predict(test.drop(['id'],axis=1))
#results=[]

#for i in predictionsxgb:

#    results.append(pred_country[i])

#print(results)
#my_submissionxgb = pd.DataFrame({'id': test.id, 'country':results})

#my_submissionxgb.to_csv('submission.csv', index=False)