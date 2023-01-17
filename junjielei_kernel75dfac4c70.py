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
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer


import numpy as np
import pandas as pd


import time
import matplotlib.pylab as plt

import gc # garbage collection
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
cal = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")
price = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")
df_stv = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

df_stv.head()

for i in range(10):
    plt.plot(df_stv.loc[df_stv['item_id'] == 'HOBBIES_1_001'].iloc[i, 6:].values,
             label=df_stv.loc[df_stv['item_id'] == 'HOBBIES_1_001'].iloc[i, 5]);
plt.title('HOBBIES_1_001 sales')
plt.legend();


plt.figure(figsize=(12, 4))
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.plot(df_stv.loc[df_stv['item_id'] == 'HOBBIES_1_001'].iloc[i, 6:].rolling(10).mean().values,
             label=df_stv.loc[df_stv['item_id'] == 'HOBBIES_1_001'].iloc[i, 5]);
plt.title('HOBBIES_1_001 sales, rolling mean 30 days')
plt.legend();
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.plot(df_stv.loc[df_stv['item_id'] == 'HOBBIES_1_001'].iloc[i, 6:].rolling(60).mean().values,
             label=df_stv.loc[df_stv['item_id'] == 'HOBBIES_1_001'].iloc[i, 5]);
plt.title('HOBBIES_1_001 sales, rolling mean 60 days')
plt.legend();
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.plot(df_stv.loc[df_stv['item_id'] == 'HOBBIES_1_001'].iloc[i, 6:].rolling(90).mean().values,
             label=df_stv.loc[df_stv['item_id'] == 'HOBBIES_1_001'].iloc[i, 5]);
plt.title('HOBBIES_1_001 sales, rolling mean 90 days')
plt.legend()
# cal.head()
cal.sample(5)
price.head()
ca_1_sales = df_stv.loc[df_stv['store_id'] == 'CA_1']
pd.crosstab(ca_1_sales['cat_id'], ca_1_sales['dept_id']).reset_index()
cal["event_type_1_snap"] = pd.notna(cal["event_type_1"]) 
cal["event_type_2_snap"] = pd.notna(cal["event_type_2"]) 
cal["date"] =  pd.to_datetime(cal["date"])
cal["d_month"] = cal["date"].dt.day
cal["year"] = pd.to_numeric(cal["year"])
cal["wday"] = pd.to_numeric(cal["wday"])
sales_data = pd.merge(price, cal[["year","month","d","wday","weekday",
                                             "event_type_1_snap","event_type_2_snap","wm_yr_wk"]], 
                      left_on='wm_yr_wk', right_on='wm_yr_wk')
clus20 = df_stv.iloc[:,2:]
column_index = [1,2,3,4,5]
for i in range(6 , len(df_stv.columns)):
    column_index.append(i)

clus_hobbies = df_stv.iloc[:,column_index].query("cat_id == 'HOBBIES'")
clus_household = df_stv.iloc[:,column_index].query("cat_id == 'HOUSEHOLD'")
clus_foods = df_stv.iloc[:,column_index].query("cat_id == 'FOODS'")
clus_ca = df_stv.iloc[:,column_index].query("state_id == 'CA'")
clus_tx = df_stv.iloc[:,column_index].query("state_id == 'TX'")
clus_wi = df_stv.iloc[:,column_index].query("state_id == 'WI'")
clus = df_stv.iloc[:,column_index]
from datetime import datetime
columnsets = []
for i in range(1,32):      
    d = cal[:1913].query("d_month == "+ str(i))["d"]
    columnsets.append([d.values])

# this chunk of code is similar to what we did in class
# the purpose it to do Label encoding for catagorical data

def label_encoding(data_preap,cat_features):
    categorical_names = {}
    data = []
    encoders = []
    
    data = data_preap[:]
    for feature in cat_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data.iloc[:,feature])
        data.iloc[:, feature] = le.transform(data.iloc[:, feature])
        categorical_names[feature] = le.classes_
        encoders.append(le)
    X_data = data.astype(float)
    return X_data, encoders
# this chunk of chunks is to define & train random forest model

def train_model(X_train, X_test, Y_train, Y_test):
    # Random forest regressor model with Training dataset
    start_time = datetime.today()
    regressor = RandomForestRegressor(n_estimators = 350, random_state = 50)
    regressor.fit(X_train,Y_train)

    # show the run time for our models
    print("Time taken to Train Model: " + str(datetime.today() - start_time))

    # Running Regession model score check
    Y_score = regressor.score(X_test,Y_test)
    return regressor,Y_score

def model_prediect(regressor,X_data):
    # Predicting model model result
    Y_pred = regressor.predict(X_data)
    return Y_pred
# generating rmse value for the model predection
# means square error and rmse

def validate_model(regressor,X_validation, Y_validation):
   
    Y_validation_pred = model_prediect(regressor, X_validation)
    mse = mean_squared_error(Y_validation, Y_validation_pred)
    rmse = np.sqrt(mse)
    return rmse, Y_validation_pred
# from pandas df --> data range

def get_data_range(Inital_Range,start_index,end_index):
    result = []
    [result.append(a) for a in Inital_Range]
    for i in range(max(Inital_Range) +1 + start_index, end_index):
        result.append(i)
    return result
## and then we wrap all the functions together into this run_prediction main function

def run_predictions(orig_data):
    process_data = orig_data[:]
    results = pd.DataFrame()
    for s in range(1,29):
        categorical_features = [0,1]
        data = []
        data_range = []
        for i in range(0,s):
            [data_range.append(a) for a in columnsets[i]]
        data_list = [process_data[a] for a in data_range]
        data  = pd.concat(data_list,axis = 1)


        data.insert(loc=0, column='item_id', value=process_data["item_id"])
        data.insert(loc=1, column='store_id', value=process_data["store_id"])
        X_data_preap = data[:]

        d = get_data_range(categorical_features,0,len(X_data_preap.columns)-1)   
        X,label_encoders = label_encoding(X_data_preap.iloc[:,d],categorical_features)
        Y = X.iloc[:,-1]

        d_validation = get_data_range(categorical_features,1,len(X_data_preap.columns))   
        X_validation,label_encoders_validation = label_encoding(X_data_preap.iloc[:,d_validation],categorical_features)
        Y_validation = X_validation.iloc[:,-1]
 
        print("Running Model for Day " + str(s))
        # Sampling data for train & split
        X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[:,0:len(X.columns)-1],Y,test_size = 0.2, random_state = 0)
        model, score = train_model(X_train, X_test, Y_train, Y_test)
        print("Model Score: " + str(score))
        
        # for the initial model
        rmse,validation_predictions = validate_model(model,X_validation.iloc[:,0:len(X_validation.columns)-1], Y_validation)
        print("RMSE Result: " + str(rmse))
        
        if (len(results.columns) == 0):
            for feature in categorical_features:
                results[feature] = label_encoders_validation[feature].inverse_transform(X_validation.iloc[:,feature].astype(int))

        results["d_" + str(s)] = validation_predictions.astype(int)
        print(results)
        results.to_csv('pd_predictions_' + str(s) +'.csv')
    return results
pd_predictions = run_predictions(clus)


