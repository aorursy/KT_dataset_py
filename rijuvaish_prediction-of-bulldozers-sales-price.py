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
import matplotlib.pyplot as plt
import sklearn
df=pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv',low_memory=False)
df.info()
df.isna().sum()
fig,ax =plt.subplots()
ax.scatter(df['saledate'][:1000],df['SalePrice'][:1000])
df.SalePrice.plot.hist()
df.columns
df=pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv',low_memory=False,parse_dates=['saledate'])
df.saledate.head()
fig,ax =plt.subplots()
ax.scatter(df['saledate'][:1000],df['SalePrice'][:1000])
df.head()
df.saledate.head(20)
df.sort_values(by='saledate',inplace=True,ascending=True)
df_temp=df.copy()
df_temp['saleYear']=df_temp.saledate.dt.year
df_temp['saleMonth']=df_temp.saledate.dt.month
df_temp['saleDay']=df_temp.saledate.dt.day
df_temp['saleDayofWeek']=df_temp.saledate.dt.dayofweek
df_temp['saleDayofYear']=df_temp.saledate.dt.dayofyear
df_temp.head().T
df_temp.drop('saledate',axis=1,inplace=True)
#find columns which contain string
for label , content in df_temp.items():
    if pd.api.types.is_string_dtype(content):
        df_temp[label]=content.astype('category').cat.as_ordered()
       
df_temp.info()
df_temp.state.cat.categories
df_temp.isnull().sum()/len(df_temp)
#fill numeric rows with median
for label , content in df_temp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_temp[label]=content.fillna(content.median())
       
#Turn categories into number and fill missing values
for label , content in df_temp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_temp[label]=pd.Categorical(content).codes +1
        
            
df_temp.isna().sum() #no  more missing values
len(df_temp)

from sklearn.ensemble import RandomForestRegressor

%%time
model=RandomForestRegressor(n_jobs=-1,random_state=42)

model.fit(df_temp.drop('SalePrice',axis=1),df_temp['SalePrice'])
#Score the model
model.score(df_temp.drop('SalePrice',axis=1),df_temp['SalePrice'])
df_val = df_temp[df_temp.saleYear==2012]
df_train = df_temp[df_temp.saleYear !=2012]
len(df_val),len(df_train)
#Split data into x and Y
X_train,y_train=df_train.drop('SalePrice',axis=1),df_train['SalePrice']
X_valid,y_valid=df_val.drop('SalePrice',axis=1),df_val['SalePrice']
X_train.shape,y_train.shape,X_valid.shape,y_valid.shape
#Creating an evalutaion function (competion uses RMSLE)
from sklearn.metrics import mean_squared_log_error,mean_absolute_error,r2_score
def rmsle(y_test,y_pred):
    """
    Calculates root mena squared log error
    """
    return np.sqrt(mean_squared_log_error(y_test,y_pred))
#Creating function to evaluate model on few different levels
def show_scores(model):
    train_preds=model.predict(X_train)
    val_preds=model.predict(X_valid)
    scores={"Training MAE":mean_absolute_error(y_train,train_preds)
            ,"Valid MaE":mean_absolute_error(y_valid,val_preds),
           "Training RMSLE":rmsle(y_train,train_preds),
           "valid RMSLE": rmsle(y_valid,val_preds),
           "Training R^2":r2_score(y_train,train_preds),
           "Valid R^2":r2_score(y_valid,val_preds)}
    return scores

#Change max samples values
model=RandomForestRegressor(n_jobs=-1,
                           random_state=42,
                           max_samples=10000)
%%time
model.fit(X_train,y_train)
show_scores(model)
%%time
from sklearn.model_selection import RandomizedSearchCV
#Different RandomForestRegressor hyberparameters
rf_grid={"n_estimators":np.arange(10,100,10),
         "max_depth":[None,3,5,10],
         "min_samples_split":np.arange(2,20,2),
         "min_samples_leaf":np.arange(1,20,2),
         "max_features":[0.5,1,"sqrt","auto"],
         "max_samples":[10000]
    
}
rs_model=RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                 random_state=42),
                                                 param_distributions=rf_grid,
                           n_iter=50,
                           cv=5,
                           verbose=True)
rs_model.fit(X_train,y_train)
rs_model.best_params_
show_scores(rs_model)
ideal_model=RandomForestRegressor(n_estimators=40,
                                 min_samples_leaf=1,
                                 min_samples_split=14,
                                 max_features=0.5,
                                 n_jobs=-1,
                                 max_samples=None)
ideal_model.fit(X_train,y_train)
#trained on all the data
show_scores(ideal_model)
df_test=pd.read_csv("/kaggle/input/bluebook-for-bulldozers/Test.csv",
                   low_memory=False,
                   parse_dates=["saledate"])
df_test.head()
test_preds=ideal_model.predict(df_test)
df_test.isna().sum()
def preprocess_data(df):
    """
    Perform trasnsormation on df and returns transformed df.
    """
    df['saleYear']=df.saledate.dt.year
    df['saleMonth']=df.saledate.dt.month
    df['saleDay']=df.saledate.dt.day
    df['saleDayofWeek']=df.saledate.dt.dayofweek
    df['saleDayofYear']=df.saledate.dt.dayofyear
    df.drop('saledate',axis=1,inplace=True)
    #find columns which contain string
    for label , content in df.items():
        if pd.api.types.is_string_dtype(content):
            df[label]=content.astype('category').cat.as_ordered()
    #fill numeric rows with median
    for label , content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                 df[label]=content.fillna(content.median())	
    #Turn categories into number and fill missing values
    for label , content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            df[label]=pd.Categorical(content).codes +1
    
    return df

preprocess_data(df_test)
df_test.isna().sum()
test_preds=ideal_model.predict(df_test)
test_preds
#formatting predictions
df_preds=pd.DataFrame()
df_preds['SalesID']=df_test['SalesID']
df_preds['SalePrice']=test_preds
df_preds
#export
df_preds.to_csv("/kaggle/working/test_predictions.csv",index=False)
