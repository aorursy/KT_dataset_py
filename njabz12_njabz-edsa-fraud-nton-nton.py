import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

from sklearn.utils import resample
train = pd.read_csv('../input/training.csv')

test = pd.read_csv('../input/test.csv')



#Save the 'id' column

Test_id = test['TransactionId']

Train_id = train['TransactionId']



#Dropping the  'id' column 

train.drop("TransactionId", axis = 1, inplace = True)

test.drop("TransactionId", axis = 1, inplace = True)

print(test.shape)

test.tail()
print(train.shape)

train.tail()
print(train.FraudResult.value_counts())
(len(train.loc[train.FraudResult==1])) / (len(train.loc[train.FraudResult == 0])) * 100
test.isnull().values.any()
def dummyEncode(df):

        columnsToEncode = list(df.select_dtypes(include=['category','object']),)

        le = LabelEncoder()

        for feature in columnsToEncode:

            try:

                df[feature] = le.fit_transform(df[feature])

            except:

                print('Error encoding '+feature)

        return df
dummyEncode(train)

dummyEncode(test)
#train.drop("TransactionStartTime", axis = 1, inplace = True)

#train.drop("BatchId", axis = 1, inplace = True)

#train.drop("SubscriptionId", axis = 1, inplace = True)

#train.drop("ProviderId", axis = 1, inplace = True)

#train.drop("ChannelId", axis = 1, inplace = True)

#train.drop("AccountId", axis = 1, inplace = True)

#train.drop("TransactionStartTime", axis = 1, inplace = True)

#train.drop("TransactionStartTime", axis = 1, inplace = True)

#train.drop("TransactionStartTime", axis = 1, inplace = True)

#train.drop("TransactionStartTime", axis = 1, inplace = True)

#train.drop("TransactionStartTime", axis = 1, inplace = True)

#train.head()
#one_hot=pd.get_dummies(train, columns=['ProductCategory'])

#train = train.drop('ProductCategory',axis = 1)

#train = train.join(one_hot,how='left')

#train.head()
test['FraudResult'] = 0



all_data = pd.concat((train, test)).reset_index(drop=True)
print(all_data.shape)

all_data.head()
X = all_data.drop('FraudResult', axis=1)

y = all_data['FraudResult']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



# Putting the data back together

X= pd.concat([X_train, y_train], axis=1)



# separate minority and majority classes

not_fraud = X[X.FraudResult==0]

fraud = X[X.FraudResult==1]



# upsample minority

fraud_upsampled = resample(fraud,

                          replace=True, # sample with replacement

                          n_samples=len(not_fraud), # match number in majority class

                          random_state=42) # reproducible results



# combine majority and upsampled minority

upsampled = pd.concat([not_fraud, fraud_upsampled])



# check new class counts

upsampled.FraudResult.value_counts()
not_fraud_downsampled = resample(not_fraud,

                                replace = False, # sample without replacement

                                n_samples = len(fraud), # match minority n

                                random_state = 27) # reproducible results

downsampled = pd.concat([not_fraud_downsampled, fraud])



# checking counts

downsampled.FraudResult.value_counts()
y_train_down = downsampled.FraudResult

X_train_down = downsampled.drop('FraudResult', axis=1)
undersampled = LogisticRegression().fit(X_train_down, y_train_down)



undersampled_pred = undersampled.predict(X_test)



# Checking accuracy

print(accuracy_score(y_test, undersampled_pred))

# f1 score

print(f1_score(y_test, undersampled_pred))



print(recall_score(y_test, undersampled_pred))
y_train = upsampled.FraudResult

X_train = upsampled.drop('FraudResult', axis=1)
y_train.head()
X_train.head()

#X_train.drop("TransactionId", axis = 1, inplace = True)
lr_upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)
lr_pred = lr_upsampled.predict(X_test)
f1_score(y_test, lr_pred)
np.prod(lr_pred.shape)
#X = all_data.drop('FraudResult', axis=1)

#y = all_data['FraudResult']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# DummyClassifier to predict only target 0

#dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

#dummy_pred = dummy.predict(X_test)



# checking unique labels

#print('Unique predicted labels: ', (np.unique(dummy_pred)))



# checking accuracy

#print('Test score: ', accuracy_score(y_test, dummy_pred))
f1_score(y_test, lr_pred)
pd.DataFrame(confusion_matrix(y_test, lr_pred))
recall_score(y_test, lr_pred)
submission = pd.DataFrame(data = Test_id, columns= ['TransactionId'])

submission['FraudResult']= pd.Series(undersampled_pred)

submission.head()
submission['FraudResult'] = submission['FraudResult'].fillna(submission['FraudResult'].mode()[0])
submission['FraudResult'] = submission['FraudResult'].astype(int)
submission.dtypes
submission.head()
submission.to_csv('3rd_try.csv', index=False)
#ntrain = train.shape[0] 

#ntest = test.shape[0]

#y_train = train[['FraudResult']]

#features = pd.concat((train, test), sort=False).reset_index(drop=True)

#features.drop(['FraudResult'], axis=1, inplace=True)

#print(features.shape)

#features.head()
#def dummyEncode(df):

        #columnsToEncode = list(df.select_dtypes(include=['category','object']),)

        #le = LabelEncoder()

        #for feature in columnsToEncode:

            #try:

                #df[feature] = le.fit_transform(df[feature])

            #except:

                #print('Error encoding '+feature)

        #return df
#le = LabelEncoder()

#all_data['BatchId']=le.fit_transform(all_data['BatchId'])

#all_data['AccountId']=le.fit_transform(all_data['AccountId'])

#all_data['SubscriptionId']=le.fit_transform(all_data['SubscriptionId'])

#all_data['CustomerId']=le.fit_transform(all_data['CustomerId'])

#all_data['CurrencyCode']=le.fit_transform(all_data['CurrencyCode'])

#all_data['ProviderId']=le.fit_transform(all_data['ProviderId'])

#all_data['ChannelId']=le.fit_transform(all_data['ChannelId'])

#all_data['TransactionStartTime']=le.fit_transform(all_data['TransactionStartTime'])

#all_data['ProductId']=le.fit_transform(all_data['ProductId'])

#all_data['ProductCategory']=le.fit_transform(all_data['ProductCategory'])
#print(features.shape)

#features.head()
#dummyEncode(features)

# Dropping Transaction ID

#features.drop("TransactionId", axis = 1, inplace = True)
#X_train = features[:ntrain]

#X_test = features[ntrain:]
#y_train
#y = train.FraudResult

#X = train.drop('FraudResult', axis=1)



# setting up testing and training sets

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

#lr_pred = lr.predict(X_test)

#print("Accuracy",accuracy_score(y_test, lr_pred))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.