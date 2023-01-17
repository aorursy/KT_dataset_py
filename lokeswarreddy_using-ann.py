# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


from pandas_profiling import ProfileReport
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sub=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train_data.head(3)
test_data.head(3)
train_data.shape
test_data.shape

train_data.info()
train_data.SalePrice
test_data.info()
df=pd.concat([train_data.iloc[:,:-1],test_data],axis=0)
df.shape
print(df.isnull().sum())
percentage =((df.isnull().sum())/len(df))
t=percentage.sort_values(ascending=False)
list_of_null=t[t>0]
list_of_null
with open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt') as f:
    print(f.read())
features=['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
df[features].info()
for col in features:

    df[col]=df[col].fillna('None')
def display_missing(df1):
    missing=(df1.isnull().sum())/(len(df1))*100
    missing=missing.drop(missing[missing==0].index).sort_values(ascending=False)
    return missing
lis=display_missing(df)
lis
def features_dtypes(d):
    t=d.index
    for i in t:
        print(i,"\t",df[i].dtype)
features_dtypes(lis)
features_1=['GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea']
for col in features_1:
    df[col]=df[col].fillna(0)
missing_1=display_missing(df)
missing_1
features_dtypes(missing_1)
features_2=[ 'MSZoning', 'Functional', 'Utilities', 'SaleType',
       'KitchenQual', 'Electrical', 'Exterior2nd', 'Exterior1st']
for col in features_2:
    df[col].fillna(df[col].mode()[0],inplace=True)
display_missing(df)
df['LotFrontage']=df.groupby("Neighborhood")['LotFrontage'].transform(lambda x: x.fillna(x.median()))
display_missing(df)
df.select_dtypes(include='object').columns
df.dtypes.value_counts()
train_df=df[:1460]

test_df=df[1460:]
train_df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True);
test_df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True);

objects = train_df.columns[train_df.dtypes == 'object'].to_list()
train_x=pd.get_dummies(train_df,columns=objects)
for i in objects:
    cols = train_x.filter(like=i).columns
    train_x.drop(cols[0],axis=1,inplace=True)
    
objects = test_df.columns[test_df.dtypes == 'object'].to_list()
test=pd.get_dummies(test_df,columns=objects)
for i in objects:
    cols = test.filter(like=i).columns
    test.drop(cols[0],axis=1,inplace=True)

missing = (list(set(train_x.columns) - set(test.columns)))
train_x.drop(columns = missing,axis = 1,inplace=True)
train_x.head()
train_x.drop('Id',axis=1,inplace = True)

train_x.shape[1]
test.drop('Id',axis=1,inplace = True)

test.shape[1]

y_new
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

train_x=scaler.fit_transform(train_x)

test=scaler.fit_transform(test)

y=train_data.iloc[:,-1]

y=scaler.fit_transform(y.values.reshape(-1,1))

y_new=scaler.inverse_transform(y.reshape(-1,1))
y_new
classifier=Sequential()
classifier.add(Dense(units=512,kernel_initializer='uniform',activation='relu',input_dim=225))

classifier.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=64,kernel_initializer='uniform',activation='relu'))

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='relu'))
classifier.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
classifier.fit(train_x,y,batch_size=15,epochs=500)

y_pred = classifier.predict(test) 


y_new_=scaler.inverse_transform(y_pred.reshape(-1,1))
pred=[]

for i in y_new_:
    pred.append(i.tolist()[0])
test_df
new_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
output = pd.DataFrame({'Id': test_df.Id,'SalePrice': pred})
output.to_csv('submission.csv', index=False)