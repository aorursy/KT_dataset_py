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
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from plotly import __version__
from plotly.offline import download_plotlyjs,init_notebook_mode,iplot,plot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
pd.options.display.float_format = '{:.2f}'.format
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df_train.head(2)
df_train.info()
def findMissingValue(df):
    data = pd.DataFrame(columns=['Columns','Number_Of_MValue','Percentage_Of_MValue'])
    
    data_col = []
    data_mValue = []
    data_perMValu = []
    
    for fn in df.columns:
        targetNum = len(df)
        x= df[fn].describe()[0]
        if x !=targetNum:
            missingValue = targetNum-x
            percentOfMV = round(float((missingValue/targetNum)*100),2)
            percentOfMV= str(percentOfMV) + ' '+ '%'
            
            data_col.append(fn)
            data_mValue.append(missingValue)
            data_perMValu.append(percentOfMV)
    
    data['Columns']=data_col
    data['Number_Of_MValue']=data_mValue
    data['Percentage_Of_MValue']=data_perMValu
    
    return data.sort_values('Number_Of_MValue',ascending=False).reset_index(drop=True)
findMissingValue(df_train)
findMissingValue(df_test)
df_train.drop(columns=['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
df_test.drop(columns=['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
for i in df_train.columns:
    if 'qu' in i.lower():
        print(i)
def convert_qual_to_num(df):
    for i in df[['FireplaceQu','OverallQual','ExterQual','BsmtQual','LowQualFinSF','KitchenQual','GarageQual']]:
        df[i].mask(df[i] == 'Ex', 5, inplace=True)
        df[i].mask(df[i] == 'Gd', 4, inplace=True)
        df[i].mask(df[i] == 'TA', 3, inplace=True)
        df[i].mask(df[i] == 'Fa', 2, inplace=True)
        df[i].mask(df[i] == 'Po', 1, inplace=True)
        df[i].mask(df[i] == 'NA', 0, inplace=True)
convert_qual_to_num(df_train)
convert_qual_to_num(df_test)
for i in range(5):
    mean_value = df_train[df_train['FireplaceQu']==(i+1)][['OverallQual','ExterQual','BsmtQual','KitchenQual','GarageQual']].sum(axis=1).unique().mean()
    print('The mean value for rating '+str(i+1)+' is - '+str(round(mean_value,2)))
def fillup_mValue_fireplace(df):
    
    mask = (df['FireplaceQu'].isna()) & df[['OverallQual','ExterQual','BsmtQual','KitchenQual','GarageQual']].sum(axis=1)>=23
    df['FireplaceQu'][mask] = 5
    mask = (df['FireplaceQu'].isna()) & df[['OverallQual','ExterQual','BsmtQual','KitchenQual','GarageQual']].sum(axis=1)>=20
    df['FireplaceQu'][mask] = 4
    mask = (df['FireplaceQu'].isna()) & df[['OverallQual','ExterQual','BsmtQual','KitchenQual','GarageQual']].sum(axis=1)>=17
    df['FireplaceQu'][mask] = 3
    mask = (df['FireplaceQu'].isna()) & df[['OverallQual','ExterQual','BsmtQual','KitchenQual','GarageQual']].sum(axis=1)>=15
    df['FireplaceQu'][mask] = 2
    mask = (df['FireplaceQu'].isna()) & df[['OverallQual','ExterQual','BsmtQual','KitchenQual','GarageQual']].sum(axis=1)>=5
    df['FireplaceQu'][mask] = 1
    mask = (df['FireplaceQu'].isna()) & df[['OverallQual','ExterQual','BsmtQual','KitchenQual','GarageQual']].sum(axis=1)<5
    df['FireplaceQu'][mask] = 0
fillup_mValue_fireplace(df_train)
fillup_mValue_fireplace(df_test)
df_train[df_train['LotFrontage'].notna()]['LotFrontage'].sum()/len(df_train[df_train['LotFrontage'].notna()])
df_test[df_test['LotFrontage'].notna()]['LotFrontage'].sum()/len(df_test[df_test['LotFrontage'].notna()])
df_train['LotFrontage'].fillna(value=70,axis=0,inplace=True)
df_test['LotFrontage'].fillna(value=70,axis=0,inplace=True)
findMissingValue(df_train)
df_train['GarageYrBlt'].fillna(value=df_train['YearBuilt'],axis=0,inplace=True)
df_test['GarageYrBlt'].fillna(value=df_test['YearBuilt'],axis=0,inplace=True)
findMissingValue(df_train)
df_train['GarageType'].fillna(value='No',axis=0,inplace=True)
df_test['GarageType'].fillna(value='No',axis=0,inplace=True)
df_train['GarageFinish'].fillna(value='NoGarage',axis=0,inplace=True)
df_test['GarageFinish'].fillna(value='NoGarage',axis=0,inplace=True)
df_train['GarageQual'].fillna(value=0,axis=0,inplace=True)
df_test['GarageQual'].fillna(value=0,axis=0,inplace=True)
df_train['GarageCond'].fillna(value='Po',axis=0,inplace=True)
df_test['GarageCond'].fillna(value='Po',axis=0,inplace=True)
df_train['BsmtExposure'].fillna(value='No',axis=0,inplace=True)
df_test['BsmtExposure'].fillna(value='No',axis=0,inplace=True)
df_train['BsmtFinType2'].fillna(value='No',axis=0,inplace=True)
df_test['BsmtFinType2'].fillna(value='No',axis=0,inplace=True)
df_train['BsmtQual'].fillna(value=0,axis=0,inplace=True)
df_test['BsmtQual'].fillna(value=0,axis=0,inplace=True)
df_train['BsmtCond'].fillna(value='Po',axis=0,inplace=True)
df_test['BsmtCond'].fillna(value='Po',axis=0,inplace=True)
df_train['BsmtFinType1'].fillna(value='No',axis=0,inplace=True)
df_test['BsmtFinType1'].fillna(value='No',axis=0,inplace=True)
df_train['MasVnrType'].fillna(value='No',axis=0,inplace=True)
df_test['MasVnrType'].fillna(value='No',axis=0,inplace=True)
df_train['MasVnrArea'].fillna(value=0,axis=0,inplace=True)
df_test['MasVnrArea'].fillna(value=0,axis=0,inplace=True)
df_train['Electrical'].fillna(value='No',axis=0,inplace=True)
df_test['Electrical'].fillna(value='No',axis=0,inplace=True)
findMissingValue(df_train)
findMissingValue(df_test)
df_test['MSZoning'].fillna(value='No',axis=0,inplace=True)
df_test['Utilities'].fillna(value='No',axis=0,inplace=True)
df_test['BsmtFullBath'].fillna(value=0,axis=0,inplace=True)
df_test['BsmtHalfBath'].fillna(value=0,axis=0,inplace=True)
df_test['Functional'].fillna(value='No',axis=0,inplace=True)
df_test['Exterior1st'].fillna(value='No',axis=0,inplace=True)
df_test['Exterior2nd'].fillna(value='No',axis=0,inplace=True)
df_test['BsmtFinSF1'].fillna(value=0,axis=0,inplace=True)
df_test['BsmtFinSF2'].fillna(value=0,axis=0,inplace=True)
df_test['BsmtUnfSF'].fillna(value=0,axis=0,inplace=True)
df_test['TotalBsmtSF'].fillna(value=0,axis=0,inplace=True)
df_test['KitchenQual'].fillna(value=0,axis=0,inplace=True)
df_test['GarageCars'].fillna(value=0,axis=0,inplace=True)
df_test['GarageArea'].fillna(value=0,axis=0,inplace=True)
df_test['SaleType'].fillna(value='No',axis=0,inplace=True)
findMissingValue(df_test)
g = sns.pairplot(df_train, vars=['SalePrice','MSSubClass','MSZoning','Neighborhood','HouseStyle','OverallQual','YearBuilt','LandContour','Utilities'])
df_train.iplot(kind='bar',x='MSSubClass',y='SalePrice',gridcolor='blue',bins=30)
df_train.iplot(kind='bar',x='MSZoning',y='SalePrice',gridcolor='blue',bins=30)
df_train.iplot(kind='bar',x='Neighborhood',y='SalePrice',gridcolor='blue',bins=30)
df_train.iplot(kind='bar',x='OverallQual',y='SalePrice',gridcolor='blue',bins=30)
df_train.iplot(kind='bar',x='YearBuilt',y='SalePrice',gridcolor='blue',bins=30)
df_train.iplot(kind='bar',x='LandContour',y='SalePrice',gridcolor='blue',bins=30)
df_train.iplot(kind='bar',x='Utilities',y='SalePrice',gridcolor='blue',bins=30)
df_test_copy = df_test.copy()
df_train.drop(columns=['Id'],axis=1,inplace=True)
df_test.drop(columns=['Id'],axis=1,inplace=True)
df_train.columns.nunique()
df_test.columns.nunique()
df_train = pd.get_dummies(df_train,drop_first=True)
df_train.columns.nunique()
df_test= pd.get_dummies(df_test,drop_first=True)
df_test.columns.nunique()
df_train.columns.difference(df_test.columns)
df_train.drop(columns=['Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn',
       'Electrical_Mix', 'Electrical_No', 'Exterior1st_ImStucc',
       'Exterior1st_Stone', 'Exterior2nd_Other', 'Heating_GasA',
       'Heating_OthW', 'HouseStyle_2.5Fin', 'KitchenQual_3', 'KitchenQual_4',
       'KitchenQual_5', 'RoofMatl_CompShg', 'RoofMatl_Membran',
       'RoofMatl_Metal', 'RoofMatl_Roll', 'Utilities_NoSeWa'],axis=1,inplace=True)
df_train.columns.nunique()
df_test.columns.difference(df_train.columns)
df_test.drop(columns=['Exterior1st_No', 'Exterior2nd_No', 'Functional_No', 'KitchenQual',
       'MSZoning_No', 'SaleType_No', 'Utilities_No'],axis=1,inplace=True)
df_test.columns.nunique()
X= df_train.drop(['SalePrice'],axis=1)
y= df_train['SalePrice']
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression

#Create an object of the classifier.
bbc_lr = BalancedBaggingClassifier(base_estimator=LogisticRegression(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=1)

y_train = df_train['SalePrice']
X_train = df_train.drop(['SalePrice'], axis=1, inplace=False)

#Train the classifier.
bbc_lr.fit(X_train, y_train)


prediction = bbc_lr.predict(X_test)

bbc_lr.score(X_test,y_test)
print('The Logistic Regression Accuracy is {:.2f} %'.format(bbc_lr.score(X_test,y_test)*100))

print('\n')

print(classification_report(y_test,prediction))
print('\n')
print(confusion_matrix(y_test,prediction))
from sklearn.tree import DecisionTreeClassifier

#Create an object of the classifier.
bbc_dt = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

y_train = df_train['SalePrice']
X_train = df_train.drop(['SalePrice'], axis=1, inplace=False)

#Train the classifier.
bbc_dt.fit(X_train, y_train)

prediction = bbc_dt.predict(X_test)

bbc_dt.score(X_test,y_test)
print('The Decision Tree Accuracy is {:.2f} %'.format(bbc_dt.score(X_test,y_test)*100))

print('\n')

print(classification_report(y_test,prediction))
print('\n')
print(confusion_matrix(y_test,prediction))
from sklearn.ensemble import RandomForestClassifier
#Create an object of the classifier.
bbc_rf = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

y_train = df_train['SalePrice']
X_train = df_train.drop(['SalePrice'], axis=1, inplace=False)

#Train the classifier.
bbc_rf.fit(X_train, y_train)

prediction = bbc_rf.predict(X_test)

bbc_rf.score(X_test,y_test)
print('The Random Forest Accuracy is {:.2f} %'.format(bbc_rf.score(X_test,y_test)*100))

print('\n')

print(classification_report(y_test,prediction))
print('\n')
print(confusion_matrix(y_test,prediction))
from sklearn.svm import SVC
#Create an object of the classifier.
bbc_sv = BalancedBaggingClassifier(base_estimator=SVC(random_state=1),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=1)

y_train = df_train['SalePrice']
X_train = df_train.drop(['SalePrice'], axis=1, inplace=False)

#Train the classifier.
bbc_sv.fit(X_train, y_train)

prediction = bbc_sv.predict(X_test)

bbc_sv.score(X_test,y_test)
print('The Support Vector Accuracy is {:.2f} %'.format(bbc_sv.score(X_test,y_test)*100))

print('\n')

print(classification_report(y_test,prediction))
print('\n')
print(confusion_matrix(y_test,prediction))
# Saving Model
import pickle
saved_model = pickle.dumps(bbc_rf)
# Load the Pickled model
bbc_rf_from_pickle = pickle.loads(saved_model)
# Using the loaded pickle model to make predictions
df_test['SalePrice']= bbc_rf_from_pickle.predict(df_test)
df_test_copy['SalePrice']=df_test['SalePrice']
df_test_copy[['Id','SalePrice']]
df_test.iplot(kind='bar',x='MSSubClass',y='SalePrice',gridcolor='blue',bins=30, colors='mediumpurple')
df_test.iplot(kind='bar',x='OverallQual',y='SalePrice',gridcolor='blue',bins=30, colors='mediumpurple')
df_test.iplot(kind='bar',x='YearBuilt',y='SalePrice',gridcolor='blue',bins=30, colors='mediumpurple')