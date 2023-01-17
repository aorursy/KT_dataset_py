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
train_data = pd.read_csv("/kaggle/input/DontGetKicked/training.csv")
test_data = pd.read_csv("/kaggle/input/DontGetKicked/test.csv")
train_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
train_data['MMRCurrentAuctionCleanPrice']
train_data['IsBadBuy'].value_counts()
train_data['Model'].value_counts()
train_data.drop('Model',axis =1, inplace=True)
test_data.drop('Model',axis =1, inplace=True)
train_data['Trim'].value_counts()
train_data.drop('Trim',axis =1, inplace=True)
test_data.drop('Trim',axis =1, inplace=True)
train_data['SubModel'].value_counts()
train_data.drop('SubModel',axis =1, inplace=True)
test_data.drop('SubModel',axis =1, inplace=True)
train_data['Color'].value_counts()
test_data['Color'].value_counts() #Pink is present in test data but isn't a part of training data
#fill null values of color column in train & test data
#there are 8 null values as shown by isnull().sum() above
train_data['Color'].fillna(value='Color_Unknown',inplace = True)
test_data['Color'].fillna(value='Color_Unknown',inplace = True)
#check if null values have been eliminated
print("Null values in color column of train data: "+str(train_data['Color'].isnull().sum()))
print("Null values in color column of test data: "+str(test_data['Color'].isnull().sum()))
train_data['Transmission'].value_counts()
test_data['Transmission'].value_counts()
#Manual in above indicates an error in uppercase as only 2 categories are present in test set
train_data['Transmission'].replace("Manual","MANUAL",inplace = True)
#check if error has been rectified
train_data['Transmission'].value_counts()
#fill null values in transmission column in train & test data
train_data['Transmission'].fillna(value = 'Transmission_unk',inplace = True)
test_data['Transmission'].fillna(value = 'Transmission_unk',inplace = True)
test_data['Transmission'].isnull().sum()
train_data['WheelTypeID'].value_counts()
#dropping continuous values
train_data.drop('WheelTypeID',axis =1, inplace=True)
test_data.drop('WheelTypeID',axis =1, inplace=True)
train_data['WheelType'].value_counts()
#fill null values in WheelType column in train & test data
train_data['WheelType'].fillna(value = 'WheelType_unk',inplace = True)
test_data['WheelType'].fillna(value = 'WheelType_unk',inplace = True)
train_data['Nationality'].value_counts()
#fill null values in Nationality column in train & test data
train_data['Nationality'].fillna(value = 'Nationality_unk',inplace = True)
test_data['Nationality'].fillna(value = 'Nationality_unk',inplace = True)
train_data['Size'].value_counts()
#fill null values in Size column in train & test data
train_data['Size'].fillna(value = 'Size_unk',inplace = True)
test_data['Size'].fillna(value = 'Size_unk',inplace = True)
train_data['TopThreeAmericanName'].value_counts()
#fill null values in train & test data
train_data['TopThreeAmericanName'].fillna(value = 'TopThreeAmericanName_unk',inplace = True)
test_data['TopThreeAmericanName'].fillna(value = 'TopThreeAmericanName_unk',inplace = True)
train_data['PRIMEUNIT'].value_counts()
#fill null values in train & test data
train_data['PRIMEUNIT'].fillna(value = 'Prime_unk',inplace = True)
test_data['PRIMEUNIT'].fillna(value = 'Prime_unk',inplace = True)
train_data['AUCGUART'].value_counts()
test_data['AUCGUART'].value_counts()
train_data['AUCGUART'].replace('AGREEN','GREEN',inplace=True)
test_data['AUCGUART'].replace('ARED','RED',inplace=True)
#fill null values in train & test data
train_data['AUCGUART'].fillna(value = 'AUC_unk',inplace = True)
test_data['AUCGUART'].fillna(value = 'AUC_unk',inplace = True)
#droping columns with continuous value which also contain null values
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
test_data.dtypes
#VehicleAge column is already present hence there's no need for PurchDate
train_data.drop('PurchDate',inplace=True,axis=1)
test_data.drop('PurchDate',inplace=True,axis=1)
train_data.dtypes
#list of columns that are not categorical in nature(object is dtype of categorical column)
not_categorical = train_data.drop(['RefId','IsBadBuy'],axis=1).columns[train_data.drop(['RefId','IsBadBuy'],axis=1).dtypes!='object']
#converting range of values in continuous columns to between 0-1
for i in not_categorical:
    maximum = np.max(train_data[i])
    train_data[i] = train_data[i]/maximum
    maximum_test = np.max(test_data[i])
    test_data[i] = test_data[i]/maximum_test
train_data[not_categorical].head()
categorical = train_data.drop(['RefId','IsBadBuy'],axis=1).columns[train_data.drop(['RefId','IsBadBuy'],axis=1).dtypes=='object']
categorical
for i in categorical:
    dummies = pd.get_dummies(train_data[i])
    dummies.columns = str(i)+'_'+dummies.columns
    train_data = pd.concat([train_data,dummies],axis=1)
    train_data.drop(i,inplace=True,axis=1)
    dummies = pd.get_dummies(test_data[i])
    dummies.columns = str(i)+'_'+dummies.columns
    test_data = pd.concat([test_data,dummies],axis=1)
    test_data.drop(i,inplace=True,axis=1)
train_data.shape
test_data.shape
for i in train_data.drop('IsBadBuy' ,axis=1).columns:
    if i not in test_data.columns:
        test_data[i]=np.zeros(len(test_data))
for i in test_data.columns:
    if i not in train_data.columns:
        train_data[i]=np.zeros(len(train_data))
train_data.shape
test_data.shape
train_data.head()
test_data.head()
#match order of coumns in train and test data
test_data = test_data[train_data.drop('IsBadBuy',axis=1).columns]
test_data.head()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X = train_data.drop(['RefId','IsBadBuy'],axis=1)
y = train_data['IsBadBuy']
X_train ,X_test,y_train,y_test = train_test_split(X,y,random_state = 32)
#check and make sure shapes of train and test data match
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
KNN_model = KNeighborsClassifier(n_neighbors = 11)
KNN_model.fit(X_train,y_train)
KNN_model.score(X_test,y_test)
predict=KNN_model.predict(test_data.drop('RefId',axis=1))
Submission=pd.DataFrame(data=predict,columns=['IsBadBuy'])
Submission.head()
Submission['RefId']=test_data['RefId']
Submission.set_index('RefId',inplace=True)
Submission.head()
Submission.to_csv('Submission.csv')
Submission.head()
