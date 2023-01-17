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
import sklearn
df = pd.read_csv('../input/bluebook-for-bulldozers/TrainAndValid.csv',low_memory=False)
df.head()
df.info()
df.isna().sum()
df.columns
fig,ax = plt.subplots()
ax.scatter(df['saledate'][:1000],df['SalePrice'][:1000])
df['SalePrice'].plot.hist()
df.saledate.dtype
# Import data again but this time parse dates
df = pd.read_csv('../input/bluebook-for-bulldozers/TrainAndValid.csv',low_memory=False,
                parse_dates=['saledate'])
df.saledate.dtype

df['saledate'][:5]
fig,ax = plt.subplots()
ax.scatter(df['saledate'][:1000],df['SalePrice'][:1000])
df.head()
df.head().T
# Sort dataframe by date order.
df.sort_values(by=['saledate'],inplace=True,ascending=True)
df.saledate.head()
# Make a copy of the original DataFrame.
df_tmp = df.copy()

df_tmp['saleYear'] = df_tmp.saledate.dt.year
df_tmp['saleMonth'] = df_tmp.saledate.dt.month
df_tmp['saleDay'] = df_tmp.saledate.dt.day
df_tmp['saleDayOfWeek'] = df_tmp.saledate.dt.dayofweek
df_tmp['saleDayOfYear'] = df_tmp.saledate.dt.dayofyear
df_tmp.head().T
#now we can remove saledate
df_tmp.drop('saledate',axis=1,inplace=True)
df_tmp.state.value_counts()
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1,
                             random_state=42)
model.fit(df_tmp.drop('SalePrice',axis=1),df_tmp['SalePrice'])
pd.api.types.is_string_dtype(df_tmp['UsageBand'])
# Find the columns which contains strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)
df_tmp.info()
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()
df_tmp.info()
df_tmp.state.cat.categories
df_tmp.state.cat.codes
# check missing data
df_tmp.isnull().sum()/len(df_tmp)
for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)
# Check for which nuemeric columns have null values:
for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
# Fill missing rows with median
for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_tmp[label+'_is_missing'] = pd.isnull(content)
            df_tmp[label] = content.fillna(content.median())
            
df_tmp.isna().sum()
# Check for columns which aren't numeric:
for label,content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)
pd.Categorical(df_tmp['state']).codes
# Turn categorical variables into numbers and fill missing
for label,content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_tmp[label+'_is_missing'] = pd.isnull(content)
        df_tmp[label] = pd.Categorical(content).codes+1
df_tmp.isnull().sum()
df_tmp.info()
model = RandomForestRegressor(n_jobs=-1,
                             random_state=42)
model.fit(df_tmp.drop('SalePrice',axis=1),df_tmp['SalePrice'])
model.score(df_tmp.drop('SalePrice',axis=1),df_tmp['SalePrice'])
df_tmp.saleYear.value_counts()
# Splitting data into training and validation sets
df_val = df_tmp[df_tmp.saleYear== 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]
len(df_val), len(df_train)
# Split into X and y
X_train,y_train = df_train.drop('SalePrice',axis=1),df_train['SalePrice']
X_valid,y_valid = df_val.drop('SalePrice',axis=1),df_val['SalePrice']
# Creating a RMSLE
from sklearn.metrics import mean_squared_log_error,mean_absolute_error,r2_score
def rmsle(y_test,y_preds):
    
    return np.sqrt(mean_squared_log_error(y_test,y_preds))
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {'Training MAE':mean_absolute_error(y_train,train_preds),
             'Valid MAE':mean_absolute_error(y_valid,val_preds),
             'Traing RMSLE':rmsle(y_train,train_preds),
             'valid Rmsle':rmsle(y_valid,val_preds),
             'Training R^2': r2_score(y_train,train_preds),
             'valid R^2':r2_score(y_valid,val_preds)}
    return scores
%%time
model = RandomForestRegressor(n_jobs=-1,
                             random_state=42)
model.fit(X_train,y_train)
model = RandomForestRegressor(n_jobs=-1,
                             random_state=42,
                              max_samples=10000,)
%%time

model.fit(X_train,y_train)
show_scores(model)
%%time
from sklearn.model_selection import RandomizedSearchCV

rf_grid ={'n_estimators':np.arange(10,100,10),
         'max_depth':[None,3,5,10],
         'min_samples_split':np.arange(2,20,2),
         'min_samples_leaf':np.arange(1,20,2),
         'max_features':[0.5,1,'sqrt','auto'],
         'max_samples':[10000]}
rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                   random_state=42,),
                                                   param_distributions=rf_grid,
                                                   n_iter=5,
                                                   cv=5,
                                                   verbose=True)
rs_model.fit(X_train,y_train)
rs_model.best_params_
show_scores(rs_model)
%%time
ideal_model = RandomForestRegressor(n_estimators=90,
                                   min_samples_split= 2,
                                    min_samples_leaf= 13,
                                   max_samples= 10000,
                                   max_features= 'sqrt',
                                   max_depth= None,
                                   n_jobs=-1)
ideal_model.fit(X_train,y_train)
show_scores(ideal_model)
df_test = pd.read_csv('../input/bluebook-for-bulldozers/Test.csv',
                     low_memory=False,
                     parse_dates=['saledate'])
df_test.head()
test_preds = ideal_model.predict(df_test)
df_test.isnull().sum()
df_test.info()
def preprocess_data(df):
    """
    Perform transformations on df and return transformed df.
    
    """
    # parse date
    df['saleYear'] = df.saledate.dt.year
    df['saleMonth'] = df.saledate.dt.month
    df['saleDay'] = df.saledate.dt.day
    df['saleDayOfWeek'] = df.saledate.dt.dayofweek
    df['saleDayOfYear'] = df.saledate.dt.dayofyear
    
    df.drop('saledate',axis=1,inplace=True)
    
    # Fill the numeric rows with median
    for label,content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label+'_is_missing']=pd.isnull(content)
                df[label] = content.fillna(content.median())
                
    # Fill categorical missing data and turn categories into numbers
    for label,content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            df[label+'_is_missing']=pd.isnull(content)
            #we add +1 to category code
            df[label] = pd.Categorical(content).codes+1
            
    
    return df
# process test data
df_test = preprocess_data(df_test)
df_test.head()
df_train.head()
# Make predictions on updated test data
test_preds = ideal_model.predict(df_test)
# We can find how the columns differ using sets
set(X_train.columns)-set(df_test.columns)
# Manually adjust df_test to have auctioneerID_is_missing column
df_test['auctioneerID_is_missing']=False
df_test
test_preds = ideal_model.predict(df_test)
# Format the predictions into the same format Kaggle is after:
df_preds = pd.DataFrame()
df_preds['SalesID'] =df_test['SalesID']
df_preds['SalesPrice'] = test_preds
df_preds

