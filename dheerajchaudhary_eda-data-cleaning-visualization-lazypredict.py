# Importing the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math
import sklearn
train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')   # importing the training data file
print(train_df.shape)
train_df.head()          # Print the top 5 values of the dataset 
test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')      # importing the test data file
test_df.head()
column_with_na = [features for features in train_df.columns if train_df[features].isnull().sum() >1]
print("Total number of features having some nan values is: ", len(column_with_na))
column_with_na
for features in column_with_na:
    print("Column", features, "have", np.round(train_df[features].isnull().mean(), 4)*100, "% NaN value")
for features in column_with_na:
    data = train_df.copy() 
    data[features] = np.where(data[features].isnull(), 1, 0)
    col = ["red", "green"]
    data.groupby(features)['SalePrice'].median().plot.bar(color = col)
    plt.title(features)
    plt.show()
    
## we can see in the output of this shell that, there's no such specific patterns we're getting. 
column_with_numerical_values = [features for features in train_df.columns if train_df[features].dtypes != 'O']
print("Total number of features with numerical values is: ", len(column_with_numerical_values))
# column_with_numerical_values
train_df[column_with_numerical_values].head()
for feature in column_with_numerical_values:
    data = train_df.copy()
    print(feature, len(data[feature].unique()))
column_discrete_values = [feature for feature in column_with_numerical_values if len(train_df[feature].unique()) <25]
print("Total number of numerical features with discrete values is: ", len(column_discrete_values))
data[column_discrete_values].head()

for feature in column_discrete_values:
    data = train_df.copy()    
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel("Sales Price")
    plt.show()
column_continuous_values = [feature for feature in column_with_numerical_values if len(train_df[feature].unique()) >=25]
print("Total number of numerical features with discrete values is: ", len(column_continuous_values))
data[column_continuous_values].head()

for feature in column_continuous_values:
    if feature != 'Id':  
        data[feature].hist(bins=25)
        plt.title(feature)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.show()
for feature in column_continuous_values:
    if feature != 'Id':  
        data = train_df.copy()
        if 0 in data[feature].unique():
            pass
        else:
            data[feature] = np.log(data[feature])
            data['SalePrice'] = np.log(data['SalePrice'])
            plt.scatter(data[feature], data['SalePrice'])
            plt.title(feature)
            plt.xlabel(feature)
            plt.ylabel("Count") 
            plt.show()
## We can see in the output of this shell that most of the plots showing linear corelation
for feature in column_continuous_values:
    if feature != 'Id':  
        data = train_df.copy()
        if 0 in data[feature].unique():
            pass
        else:
            data[feature] = np.log(data[feature])
#             data['SalePrice'] = np.log(data['SalePrice'])
            data.boxplot(column = feature)
            plt.title(feature)
#             plt.xlabel(feature)
            plt.ylabel(feature) 
            plt.show()
column_with_categorical_values = [features for features in train_df.columns if train_df[features].dtypes == 'O']


print("Total number of features with numerical values is: ", len(column_with_categorical_values))
# column_with_categorical_values
train_df[column_with_categorical_values].head()
for feature in column_with_categorical_values:
    data = train_df.copy()
    print(feature, len(data[feature].unique()))
for feature in column_with_categorical_values:
    data = train_df.copy()    
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel("Sales Price")
    plt.show()
column_with_year = [feature for feature in column_with_numerical_values if 'Yr' in feature or 'Year' in feature]
print("Total number of features with date entries is: ", len(column_with_year))
train_df[column_with_year]
train_df.groupby('YrSold')['SalePrice'].median().plot()
plt.title('Median Sale Price vs Sold Year')
plt.ylabel('Sale price')
plt.xlabel('Year Sold')
for feature in column_with_year:
    if feature != 'YrSold':
        data = train_df.copy()
        data[feature] = data['YrSold'] - data[feature]
        plt.scatter(data[feature], data['SalePrice'])
        plt.title("Year features vs Sales price")
        plt.xlabel(feature)
        plt.ylabel("Sale Price")
        plt.show()
# We can see if the year gap is more the price is less
test_df['SalePrice'] = 0
print("The shape of given test data is: ", test_df.shape)
## we'll concatenate both table
full_df_feature_eng = pd.concat([train_df, test_df], axis=0,sort=False)
print("The shape of dataset after combining both test and train dataset is: ",full_df_feature_eng.shape)
full_df_feature_eng.tail()   # Print the tail end of the combined datsset
columns_nan_in_categorical =[features for features in full_df_feature_eng.columns if full_df_feature_eng[features].isnull().sum()>1 and full_df_feature_eng[features].dtypes=='O']
print("Total number of categorical features having some nan values is: ", len(columns_nan_in_categorical), "\n")
for features in columns_nan_in_categorical:
    print("Column", features, "have", np.round(full_df_feature_eng[features].isnull().mean(), 4)*100, "% NaN value")

full_df_feature_eng[columns_nan_in_categorical].head()  # Printing all the categorical columns with Nan values
for features in columns_nan_in_categorical:
    full_df_feature_eng[features] = full_df_feature_eng[features].fillna('Missing')

full_df_feature_eng[columns_nan_in_categorical].head()
columns_nan_in_numerical =[features for features in full_df_feature_eng.columns if full_df_feature_eng[features].isnull().sum()>1 and full_df_feature_eng[features].dtypes!='O']
print("Total number of numerical features having some nan values is: ", len(columns_nan_in_numerical), "\n")

for features in columns_nan_in_numerical:
    print("Column", features, "have", np.round(full_df_feature_eng[features].isnull().mean(), 4)*100, "% NaN value")
full_df_feature_eng[columns_nan_in_numerical].head()   # Printing all the numerical columns with Nan values
for features in columns_nan_in_numerical:
    full_df_feature_eng[features+'Nan'] = np.where(full_df_feature_eng[features].isnull(), 1, 0) 
    full_df_feature_eng[features].fillna(full_df_feature_eng[features].median(), inplace = True)
    
for features in columns_nan_in_numerical:
    print("Column", features, "have", np.round(full_df_feature_eng[features].isnull().mean(), 4)*100, "% NaN value")

full_df_feature_eng.head()      ## we'll print the dataframe after adding 5 new columns that had nan value
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    full_df_feature_eng[feature]=np.log(full_df_feature_eng[feature])
categorical_column_in_full_data = [feature for feature in full_df_feature_eng.columns if full_df_feature_eng[feature].dtype=='O']
print(categorical_column_in_full_data)
for features in categorical_column_in_full_data:
    labels_encode=full_df_feature_eng.groupby([features])['SalePrice'].mean().sort_values().index
    labels_encode={k:i for i,k in enumerate(labels_encode,0)}
    full_df_feature_eng[features]=full_df_feature_eng[features].map(labels_encode)
full_df_feature_eng.head()
full_df_feature_eng.info()   ## Print all the columns, we can see there's no null values now
independent_feature = [features for features in full_df_feature_eng.columns if features not in ['Id', 'SalePrice']]
print(independent_feature)
scaler = MinMaxScaler()
scaler.fit(full_df_feature_eng[independent_feature])
scaler.transform(full_df_feature_eng[independent_feature])
data_scaled_independent =  pd.DataFrame(scaler.transform(full_df_feature_eng[independent_feature]), columns=independent_feature)
data_scaled_independent
concat_data = pd.concat([full_df_feature_eng[['Id', 'SalePrice']].reset_index(drop = True), pd.DataFrame(scaler.transform(full_df_feature_eng[independent_feature]), columns=independent_feature)], axis = 1)
concat_data.head()

given_train = concat_data[0:1460]
given_test = concat_data[1460:2919]
##  We'll 
X = given_train.drop(['SalePrice',
                              'Id'], axis = 1) 
target = given_train['SalePrice']
print("Dependent Variables")
display(X.head())
print("Independent Variable")
display(target.to_frame().head())
! pip install lazypredict             # install Lazypredict
# split data
from lazypredict.Supervised import LazyRegressor
X_train, X_test, Y_train, Y_test = train_test_split(X, target,test_size=.3,random_state =23)
regr=LazyRegressor(verbose=0,predictions=True)

import time
start_time_2=time.time()
models_r,predictions_r=regr.fit(X_train, X_test, Y_train, Y_test)
end_time_2=time.time()
models_r    ## Shows performance table by all the models
predictions_r  ## Print the predicted values from all the models
submission_train = given_test.drop(['SalePrice','Id'], axis = 1)       ## Drop the two columns that we added while feature engineering in concatenated file
submission_train
xgb =XGBRegressor( booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.6, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=1.5, n_estimators=2400,
             n_jobs=1, nthread=None, objective='reg:linear',
             reg_alpha=0.6, reg_lambda=0.6, scale_pos_weight=1, 
             silent=None, subsample=0.8, verbosity=1)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, target,test_size=.15,random_state =123)
#Fitting
xgb.fit(X_Train, Y_Train)
xgb_predict_sample = xgb.predict(X_Test)
print('RMSE = ' + str(math.sqrt(metrics.mean_squared_error(Y_Test, xgb_predict_sample))))          ## print the error 
lgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=3,
                                       learning_rate=0.006, 
                                       n_estimators=12000, 
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.4,   )
lgbm.fit(X_Train, Y_Train,eval_metric='rmse')
lgbm_predict_sample = lgbm.predict(X_Test)
print('RMSE = ' + str(math.sqrt(metrics.mean_squared_error(Y_Test, lgbm_predict_sample))))       ## Print the error

xgb.fit(X, target)   ## fiting xgb model
lgbm.fit(X, target,eval_metric='rmse')    ## fitting lgbm model
lgbm_predict_on_provided_test_data = lgbm.predict(submission_train)     
xgb_predict_on_provided_test_data = xgb.predict(submission_train)
req_prediction = ( lgbm_predict_on_provided_test_data*0.5 + xgb_predict_on_provided_test_data * 0.5)
antilog_of_prediction = np.exp(req_prediction)
final_prediction = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": antilog_of_prediction
    })

print("Shape of final dataframe is: ", final_prediction.shape)
final_prediction.head()                                                 
final_prediction.to_csv('submission.csv', index=False)