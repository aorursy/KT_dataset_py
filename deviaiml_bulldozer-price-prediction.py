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
import matplotlib.pyplot as plt
%matplotlib inline
filepath='../input/bluebook-for-bulldozers/'
df = pd.read_csv(filepath+'TrainAndValid.csv', low_memory=False, parse_dates=['saledate'])
df.head()
df.info()
df.isna().sum()
fig, ax = plt.subplots()
ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000]);
df.SalePrice.plot.hist();
df.head().T


# sort df by saledate

df.sort_values(by=['saledate'], inplace=True)
df.saledate.head(20)
dftemp=df.copy()
dftemp.head()
dftemp['saleyear']=dftemp['saledate'].dt.year
dftemp['salemonth']=dftemp['saledate'].dt.month
dftemp['saleday']=dftemp['saledate'].dt.day
dftemp['saledayofweek']=dftemp['saledate'].dt.dayofweek
dftemp['saledayofyear']=dftemp['saledate'].dt.dayofyear
dftemp.drop(['saledate'], inplace=True, axis=1)
dftemp.head().T
dftemp.state.value_counts()
##  split data into train and validation sets
from sklearn.model_selection import train_test_split

xtrain,  xval, ytrain, yval = train_test_split(dftemp.drop(['SalePrice'],axis=1), dftemp['SalePrice'])
print(xtrain.shape, ytrain.shape)
print(xval.shape, yval.shape)
## convert strings to numerical category values --> in xtrain, xval

for label, content in xtrain.items():
    if pd.api.types.is_string_dtype(content):
        xtrain[label]=content.astype('category').cat.as_ordered()
        
for label, content in xval.items():
    if pd.api.types.is_string_dtype(content):
        xval[label]=content.astype('category').cat.as_ordered()
xtrain.info()
xval.info()
xtrain.state.cat.categories
xtrain.state.cat.codes
xval.state.cat.categories
xval.state.cat.codes
## check missing values  --> in xtrain, xval

for label, content in xtrain.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
            
for label, content in xval.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
## fill missing values with median  --> in xtrain, xval

for label, content in xtrain.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            xtrain[label+'_is_missing']=pd.isnull(content)
            xtrain[label]=content.fillna(content.median())
            
for label, content in xval.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            xval[label+'_is_missing']=pd.isnull(content)
            xval[label]=content.fillna(content.median())
xtrain.head()
xval.head()
# check if any numeric missing values  --> xtrain, xval

for lable, content in xtrain.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
            
for lable, content in xval.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
xtrain['auctioneerID_is_missing'].value_counts()
xtrain.isna().sum()
# changing categorical values into numbers  --> xtrain, xval

for label, content in xtrain.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)
print("-----------------------------------------------")        
for label, content in xval.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)
pd.Categorical(xtrain['state']).dtype

pd.Categorical(xval['state']).dtype
pd.Categorical(xtrain['state']).codes
pd.Categorical(xval['state']).codes
# turning categories into numbers and also filling the missing values  with 0 (as missing cat values would be -1)
##  --> xtrain, xval

for label, content in xtrain.items():
    if not pd.api.types.is_numeric_dtype(content):
        xtrain[label+"_is_missing"]=pd.isnull(content)
        xtrain[label]=pd.Categorical(content).codes+1
        
        
for label, content in xval.items():
    if not pd.api.types.is_numeric_dtype(content):
        xval[label+"_is_missing"]=pd.isnull(content)
        xval[label]=pd.Categorical(content).codes+1
xtrain.isna().sum()
xval.isna().sum()
xtrain.head().T
xval.head().T
%%time
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_jobs=-1, random_state=42)
model.fit(xtrain, ytrain)
model.score(xval, yval)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
def rmsle(ytest, ypreds):
    return np.sqrt(mean_squared_log_error(ytest, ypreds))

print('RMSLE on train data : ', rmsle(ytrain, model.predict(xtrain)))
print()
print('RMSLE on validation data: ', rmsle(yval, model.predict(xval)))
## Test data predictions

# load test data

dftest = pd.read_csv(filepath+'Test.csv', low_memory=False, parse_dates=['saledate'])

dftest.shape
dftest['saleyear']=dftest['saledate'].dt.year
dftest['salemonth']=dftest['saledate'].dt.month
dftest['saleday']=dftest['saledate'].dt.day
dftest['saledayofweek']=dftest['saledate'].dt.dayofweek
dftest['saledayofyear']=dftest['saledate'].dt.dayofyear

dftest.drop(['saledate'], axis=1, inplace=True)
dftest.shape
# convert strings to categorical values
for label, content in dftest.items():
    if pd.api.types.is_string_dtype(content):
        dftest[label]=content.astype('category').cat.as_ordered()

dftest.state.cat.categories
## categories to numbers conversion and also filling missing values with 0  

for label, content in dftest.items():
    if not pd.api.types.is_numeric_dtype(content):
        dftest[label+'_is_missing']=pd.isnull(content)
        dftest[label]=pd.Categorical(content).codes+1
dftest.shape
## filling missing values with median for numeric values

for label, content in dftest.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            dftest[label+'_is_missing']=pd.isnull(content)
            dftest[label]=content.fillna(content.median())
dftest.shape
set(xtrain.columns)-set(dftest.columns)
dftest['auctioneerID_is_missing']=False
dftest.shape
testpreds=model.predict(dftest)
len(testpreds)
testpreds
dfpreds=pd.DataFrame()
dfpreds['SalesID']=dftest['SalesID']
dfpreds['SalesPrice']=testpreds
dfpreds
dfpreds.to_csv('/kaggle/working/test_predictions.csv' , index=False)