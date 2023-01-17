## Importing the data and preparing it for model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("../input/bluebook-for-bulldozers/TrainAndValid.csv",
                 low_memory=False,
                error_bad_lines=False)
df.shape
df.info()
fig , ax = plt.subplots(figsize = (20,10))
ax.scatter(df['saledate'][:1000],df['SalePrice'][:1000])
df.SalePrice.plot.hist()
df = pd.read_csv('../input/bluebook-for-bulldozers/TrainAndValid.csv', 
                 low_memory=False,
                 parse_dates=['saledate']
                )

df.info()
fig , ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])
df.head().T
df.saledate.head(10)
df.sort_values(by=['saledate'],ascending = True,inplace = True)
df.saledate.head(10)
df_tmp = df.copy()
df_tmp['saleyear'] = df_tmp.saledate.dt.year
df_tmp['salemonth'] = df_tmp.saledate.dt.month
df_tmp['saleday'] = df_tmp.saledate.dt.day
df_tmp['saledayofweek'] = df_tmp.saledate.dt.dayofweek
df_tmp['saledayofyear'] = df_tmp.saledate.dt.dayofyear

#dropping the original saledate column
df_tmp.drop('saledate',axis=1,inplace = True)
df_tmp.head().T
df_tmp.dtypes
# To check whether a column is string we use 

pd.api.types.is_string_dtype(df_tmp['UsageBand'])
# Thse columns contains string

for label , content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)
### To change all the string columns into categorical

for label , content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()
df_tmp.info()
df_tmp.state.cat.categories
df_tmp.state.cat.codes
df_tmp.isna().sum()
for label , content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)
for label , content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
for label , content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            #Adding a binary column which tells if the data is missing or not
            df_tmp[label+'_is_missing'] = pd.isnull(content)
            #Filling the numeric place with the median value
            df_tmp[label] = content.fillna(content.median())
for label , content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to inidicate whether sample had missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # We add the +1 because pandas encodes missing categories as -1
        df_tmp[label] = pd.Categorical(content).codes+1
df_tmp.info()
len(df_tmp)
df_tmp.shape
df_tmp.saleyear
df_tmp.saleyear.value_counts()
df_val = df_tmp[df_tmp.saleyear == 2012]
df_train = df_tmp[df_tmp.saleyear != 2012]

len(df_train) , len(df_val)
X_train , y_train = df_train.drop('SalePrice',axis = 1),df_train['SalePrice']
X_val , y_val = df_val.drop('SalePrice',axis = 1),df_val['SalePrice']

X_train.shape , y_train.shape , X_val.shape , y_val.shape
from sklearn.metrics import mean_squared_log_error , mean_absolute_error,r2_score


#Function to return the RMSLE

def rmsle(y_test , y_preds):
    """
    Caculates Root mean squared log error for given y_true and y_preds
    """
    return np.sqrt(mean_squared_log_error(y_test,y_preds))


#Function to evaluate model on different metrics

def show_scores(model):
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    scores = {'Training MAE':mean_absolute_error(y_train,train_preds),
              'Validation MAE':mean_absolute_error(y_val,val_preds),
              'Training RMSLE':rmsle(y_train,train_preds),
              'Validation RMSLE':rmsle(y_val,val_preds),
              'Training R2':r2_score(y_train,train_preds),
              'Validation R2':r2_score(y_val,val_preds)
             }
    

    return scores

len(X_train)
%%time

model = RandomForestRegressor(random_state=42,
                              max_samples=10000)

model.fit(X_train,y_train)
show_scores(model)
%%time

from sklearn.model_selection import RandomizedSearchCV

rf_grid = {'n_estimators':np.arange(10,100,10),
           'max_depth':[None,3,5,10],
           'min_samples_split': np.arange(2,20,2),
           'min_samples_leaf': np.arange(1,20,2),
           'max_features': [0.5,1,'sqrt','auto'],
           'max_samples' : [10000]
          }

rs_model = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                             param_distributions=rf_grid,
                             n_iter=10,
                             cv = 5,
                             verbose=True
                             )

rs_model.fit(X_train,y_train)
show_scores(rs_model)
rs_model.best_params_
%%time

ideal_model = RandomForestRegressor( n_estimators= 60,
                                     min_samples_split= 10,
                                     min_samples_leaf= 1,
                                     max_features= 'auto',
                                     max_depth= 10,
                                    random_state = 42)

ideal_model.fit(X_train,y_train)
show_scores(ideal_model)
df_test = pd.read_csv('../input/bluebook-for-bulldozers/Test.csv',
                      low_memory=False,
                      parse_dates=['saledate']
                     )
df_test.head()
df_test.isna().sum()
df_test.dtypes
def preprocess_data(df):
    """
    perform the transformations on the data and returns it
    """
    df['saleyear'] = df.saledate.dt.year
    df['salemonth'] = df.saledate.dt.month
    df['saleday'] = df.saledate.dt.day
    df['saledayofweek'] = df.saledate.dt.dayofweek
    df['saledayofyear'] = df.saledate.dt.dayofyear
    
    df.drop('saledate',axis=1,inplace = True)
    
    #Filling the numeric rows with median    
    for label , content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label+'_is_missing'] = pd.isnull(content)
                
                df[label] = content.fillna(content.median())
                
        #Filling the categorical missing data and turn categories into numeric
        if not pd.api.types.is_numeric_dtype(content):
            df[label+'_is_missing'] = pd.isnull(content)
            
            df[label] = pd.Categorical(content).codes+1
            
    return df
df_test = preprocess_data(df_test)
df_test.head()
X_train.shape , df_test.shape
set(X_train.columns) - set(df_test.columns)
df_test['auctioneerID_is_missing'] = False

df_test.shape
test_preds = ideal_model.predict(df_test)
test_preds
# Format predictions into the same format Kaggle is after
df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds
df_preds.to_csv('submission1.csv',index=False)
ideal_model.feature_importances_
#Funtion for plotting feature importances
def plot_features(columns , importances , n=20):
    """
    To plot the important features that makes the prediction
    """
    df = (pd.DataFrame({'features':columns,
                        'feature_importances':importances})
                     .sort_values('feature_importances',ascending=False)
                     .reset_index(drop=True))
    
    fig,ax = plt.subplots()
    ax.barh(df['features'][:n] , df['feature_importances'][:20])
    ax.set_ylabel('Features')
    ax.set_xlabel('Feature importance')
    ax.invert_yaxis()
plot_features(X_train.columns , ideal_model.feature_importances_)