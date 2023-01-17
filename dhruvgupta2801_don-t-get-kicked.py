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
train_data=pd.read_csv('/kaggle/input/DontGetKicked/training.csv')

train_data.head()
test_data=pd.read_csv('/kaggle/input/DontGetKicked/test.csv')

test_data.head()
print("Length of train data "+str(len(train_data)))

print("Length of test data "+str(len(test_data)))
train_data.info()
test_data.info()
train_data.isnull().sum()
test_data.isnull().sum()
train_data['IsBadBuy'].value_counts()
train_data['Model'].value_counts()
train_data.drop('Model',axis=1,inplace=True)

test_data.drop("Model",axis=1,inplace=True)
train_data['Trim'].value_counts()
train_data.drop('Trim',inplace=True,axis=1)

test_data.drop('Trim',inplace=True,axis=1)
train_data['SubModel'].value_counts()
train_data.drop('SubModel',inplace=True,axis=1)

test_data.drop('SubModel',inplace=True,axis=1)
train_data['Color'].value_counts()
test_data['Color'].value_counts()
train_data['Color'].fillna(value='Color_Unknown',inplace=True)

test_data['Color'].fillna(value='Color_Unknown',inplace=True)
print("Number of null values in Color column "+str(train_data['Color'].isnull().sum()))

print("Number of null values in Color column "+str(test_data['Color'].isnull().sum()))
train_data['Transmission'].value_counts()
test_data['Transmission'].value_counts()
train_data[train_data['Transmission']=='Manual']
train_data['Transmission'].replace("Manual","MANUAL",inplace=True)
train_data['Transmission'].value_counts()
train_data['Transmission'].fillna(value="Transmission_unk",inplace=True)

test_data['Transmission'].fillna(value="Transmission_unk",inplace=True)
train_data['WheelTypeID'].value_counts()
train_data.drop('WheelTypeID',inplace=True,axis=1)

test_data.drop('WheelTypeID',inplace=True,axis=1)
train_data['WheelType'].value_counts()
test_data['WheelType'].value_counts()
train_data['WheelType'].fillna(value='WheelType_unk',inplace=True)

test_data['WheelType'].fillna(value='WheelType_unk',inplace=True)
train_data['WheelType'].value_counts()
train_data['Nationality'].value_counts()
test_data['Nationality'].value_counts()
train_data['Nationality'].fillna(value='Nationality_unk',inplace=True)

test_data['Nationality'].fillna(value='Nationality_unk',inplace=True)
train_data['Size'].value_counts()
test_data['Size'].value_counts()
train_data['Size'].fillna(value='Size_unk',inplace=True)

test_data['Size'].fillna(value="Size_unk",inplace=True)
train_data['TopThreeAmericanName'].value_counts()
test_data['TopThreeAmericanName'].value_counts()
train_data['TopThreeAmericanName'].fillna(value='Top_unk',inplace=True)

test_data['TopThreeAmericanName'].fillna(value='Top_unk',inplace=True)
train_data['PRIMEUNIT'].value_counts()
test_data['PRIMEUNIT'].value_counts()
train_data['PRIMEUNIT'].fillna(value="Prime_unk",inplace=True)

test_data['PRIMEUNIT'].fillna(value="Prime_unk",inplace=True)
train_data['AUCGUART'].replace("AGREEN","GREEN",inplace=True)

test_data['AUCGUART'].replace("ARED","RED",inplace=True)
train_data['AUCGUART'].fillna(value="AUC_unk",inplace=True)

test_data['AUCGUART'].fillna(value="AUC_unk",inplace=True)
train_data.drop(['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',

                'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',

                'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',

                'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice'],

               inplace=True,axis=1)

test_data.drop(['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',

                'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',

                'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',

                'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice'],

               inplace=True,axis=1)
train_data.drop('PurchDate',axis=1,inplace=True)

test_data.drop("PurchDate",axis=1,inplace=True)
train_data.shape
test_data.shape
train_data.dtypes
train_data.columns
train_data.drop(['RefId','IsBadBuy'],axis=1).dtypes!='object'
not_categorical=train_data.drop(['RefId','IsBadBuy'],axis=1).columns[train_data.drop(['RefId','IsBadBuy'],axis=1).dtypes!='object']
for i in not_categorical:

    maximum=np.max(train_data[i])

    train_data[i]=train_data[i]/maximum

    maximum_test=np.max(test_data[i])

    test_data[i]=test_data[i]/maximum_test
train_data[not_categorical].head()
test_data[not_categorical].head()
categorical=train_data.drop(['RefId','IsBadBuy'],axis=1).columns[train_data.drop(['RefId','IsBadBuy'],axis=1).dtypes=='object']
categorical
train_data[categorical[0]]
pd.get_dummies(train_data[categorical[0]])
for i in categorical:

    dummies=pd.get_dummies(train_data[i])

    dummies.columns=str(i)+'_'+dummies.columns

    train_data=pd.concat([train_data,dummies],axis=1)

    train_data.drop(i,inplace=True,axis=1)

    dummies=pd.get_dummies(test_data[i])

    dummies.columns=str(i)+'_'+dummies.columns

    test_data=pd.concat([test_data,dummies],axis=1)

    test_data.drop(i,inplace=True,axis=1)    
train_data.head()
train_data.shape
test_data.shape
for i in train_data.drop('IsBadBuy',axis=1).columns:

    if i not in test_data.columns:

        test_data[i]=np.zeros(len(test_data))
for i in test_data.columns:

    if i not in train_data.columns:

        train_data[i]=np.zeros(len(train_data))
train_data.shape
test_data.shape
train_data.head()
test_data.head()
test_data=test_data[train_data.drop("IsBadBuy",axis=1).columns]
print(train_data.shape)

print(test_data.shape)
train_data.columns
test_data.columns
from sklearn.model_selection import train_test_split

X=train_data.drop(['RefId','IsBadBuy'],axis=1)

y=train_data['IsBadBuy']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=11)

KNN.fit(X_train,y_train)
KNN.score(X_test,y_test)
predict=KNN.predict(test_data.drop('RefId',axis=1))
Submission=pd.DataFrame(data=predict,columns=['IsBadBuy'])

Submission.head()
Submission['RefId']=test_data['RefId']

Submission.set_index('RefId',inplace=True)
Submission.head()

Submission.to_csv('Submission.csv')