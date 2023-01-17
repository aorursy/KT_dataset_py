# Importing required libraries

import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport

import seaborn as sns

import matplotlib.pyplot as plt

import warnings



# Setting options

np.set_printoptions(precision=4)                 # To display values upto Four decimal places. 

plt.style.use('seaborn-whitegrid')               # To apply seaborn whitegrid style to the plots.

plt.rc('figure', figsize=(20, 12))               # Set the default figure size of plots.

sns.set(style='whitegrid')                       # To apply whitegrid style to the plots.

warnings.filterwarnings('ignore')                # To ignore warnings, if any
# Importing the dataset as data

data = pd.read_csv('../input/avocado-prices/avocado.csv', index_col=0)

data.sample(8)    # Preview of random 8 rows 
data.shape       # Number of (records, features) of data
data.info()      # Info of data
data.describe()     # Descriptive statistics of data
pre_profile = data.profile_report(title='Avacado Pre-Profiling')   # Performing Pre Profiling on data.
pre_profile.to_file('pre-profiling.html')                          # Saving report to pre-profiling.html
# pre_profile.to_notebook_iframe()                                 # Displaying the profiling report inline. 
# Unique values in data index - doing this as profiling shows there are zeros in index

print('No. of unique index values:', data.index.nunique())                          
data.reset_index(drop=True, inplace=True)    # reseting index as index values seems to be incorrect



# Unique values in data index after ressetting

print('No. of unique index values after resetting index: ', data.index.nunique())   
# Renaming column names

data.rename(columns={'4046':'PLU_4046','4225':'PLU_4225','4770':'PLU_4770'}, inplace=True) # Renaming size as per description

# Renaming columns to remove spaces and capitalize first letter

data.columns = data.columns.str.replace(' ','').map(lambda x : x[0].upper() + x[1:]) 

data.head(2)  # Preview of column header
data.dtypes # Looking for data types
data['Date'] = pd.to_datetime(data['Date'])    # Converting date to datetime type

data['Year'] = data['Year'].astype('object')   # Converting Year to object from numeric
# Utility / Helper Function - To categorize season based on date



def categorizing_seasons(date):

    month = date.month



    # Source - https://en.wikipedia.org/wiki/Season#Meteorological

    winter, spring, summer, autumn = ([12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11])

    if month in winter:

        return 'Winter'

    elif month in spring:

        return 'Spring'

    elif month in summer:

        return 'Summer'

    else:

        return 'Autumn'
data['Month'] = data['Date'].dt.month_name()             # Deriving Month from Date

data['Quarter'] = data['Date'].dt.quarter                # Deriving Qurter from Date

data['Season'] = data['Date'].map(categorizing_seasons)  # Deriving Season from Date
# Utility / Helper Function - To update the variables as per data



def get_variables_from_data():

    # Target Variables

    y_column = 'AveragePrice'                                          

     

    # Categorical Feature variables 

    X_columns_cat = list(data.dtypes[data.dtypes.values == 'object'].index)  



    # Numeric Feature variables

    X_columns_num = list(data.dtypes[(data.dtypes.values != 'object') & (data.dtypes.index != y_column)].index)    



    # Feature variables

    X_columns = X_columns_num + X_columns_cat

    

    print('y_column:', y_column)

    print('X_columns: ',X_columns) 

    print('X_columns_num: ',X_columns_num) 

    print('X_columns_cat: ',X_columns_cat) 

    

    # Returning as a tuple

    return y_column, X_columns, X_columns_num, X_columns_cat
# Updating Variables

y_column, X_columns, X_columns_num, X_columns_cat = get_variables_from_data()
data.groupby('Month')[y_column].agg(['max', 'mean', 'min'])   # Understanding TV w.r.t 'Month'
data.groupby('Quarter')[y_column].agg(['max', 'mean', 'min'])  # Understanding TV w.r.t 'Quarter'
data.groupby('Season')[y_column].agg(['max', 'mean', 'min'])  # Understanding TV w.r.t 'Season'
f, ax = plt.subplots(1, 3, figsize=(15,5))

f.suptitle('Spread of mean AveragePrice Over Season, Quarter and Month', fontsize=16)

data.groupby('Season')[y_column].mean().plot(kind='bar',ax=ax[0])

data.groupby('Quarter')[y_column].mean().plot(kind='bar',ax=ax[1])

data.groupby('Month')[y_column].mean().plot(kind='bar',ax=ax[2])
X_columns          # Preview of existing Feature columns 
# Replacing date with less cordinal column month

data.drop(columns=['Date', 'Season', 'Quarter'], inplace=True)   # Droping Data, Quarter and Season columns
# Updating Variables

y_column, X_columns, X_columns_num, X_columns_cat = get_variables_from_data()
f, ax =  plt.subplots(1, 2, figsize=(15, 8))

f.suptitle('Box plot on Target Variable and Target Variable Distribution - Before', fontsize=16)

sns.boxplot(y=y_column, data=data, ax=ax[0]) # Box plot on TV before droping extreme values

sns.distplot(data[y_column], ax=ax[1])       # Distribution of Target Vaiable
# Checking mean|median and limiting data to 2 * (mean|median) - To eliminate extreme right values

data[y_column].describe()                  
data.drop(data[data[y_column] > 2.8].index, inplace=True) # Droping records where price > 3

print(data.shape)                                         # Shape of data after droping few records

data.sample(5)                                            # Preview of data after droping few records
f, ax =  plt.subplots(1, 2, figsize=(15, 8))

f.suptitle('Box plot on Target Variable and Target Variable Distribution - After', fontsize=16)

sns.boxplot(y=y_column, data=data, ax=ax[0]) # Box plot on TV after droping extreme values

sns.distplot(data[y_column], ax=ax[1])       # Distribution of Target Vaiable
data.head()   # Preview of data
# Density of mean price w.r.t categorical columns

f, ax = plt.subplots(2,2)

for x_var, subplot in zip(X_columns_cat, ax.flatten()):

    subplot.set_xlabel(x_var)

    data.groupby(x_var)[y_column].mean().plot(kind='kde', ax=subplot, label='Test')
# Mean price w.r.t categorical columns

f, ax = plt.subplots(2,2)

plt.subplots_adjust(hspace=0.5)

for x_var, subplot in zip(X_columns_cat, ax.flatten()):

    subplot.set_xlabel(x_var)

    subplot.set_ylabel('Mean Avg price')

    data.groupby(x_var)[y_column].mean().plot(kind='bar', ax=subplot, label='Test')
# Bot plot to check outliers in categorical columns



f, ax = plt.subplots(1,2, figsize=(15,5))

for x_var, subplot in zip(X_columns_cat[0:2], ax.flatten()):

    sns.boxplot(data = data, x=x_var, y=y_column, ax=subplot)



f, ax = plt.subplots(1, figsize=(15,5))

sns.boxplot(data = data, x=X_columns_cat[-1], y=y_column, ax=ax)



f, ax = plt.subplots(1, figsize=(15,5))

sns.boxplot(data = data, x=X_columns_cat[2], y=y_column, ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()
# Checking for relation of numeric columns w.r.t Target Variable

f, ax = plt.subplots(1, len(X_columns_num), figsize=(20, 5))



for x_var, sp in zip(X_columns_num, ax.flatten()):

    sns.regplot(x=data[x_var], y=data[y_column], ax=sp)
# Heatmap to check correlation

plt.figure(figsize=(10,8))

sns.heatmap(data[X_columns_num].corr(), annot=True, cmap='viridis')
# Droping highly correlated columns

data.drop(columns=['PLU_4046', 'PLU_4225', 'TotalBags', 'SmallBags'], inplace=True)
data.head(2)   # Preview after droping columns
# Updating Variables

y_column, X_columns, X_columns_num, X_columns_cat = get_variables_from_data()
sns.distplot(data[y_column]) # Normal Distribution of Target Vaiable
# Pair Plot of data

sns.pairplot(data, size = 2, aspect = 1.5)
# Checking for relation of Numeric Features with Target Variable

sns.pairplot(data, x_vars=X_columns_num, y_vars=y_column, size=5, aspect=1, kind='reg') 
post_profile = data.profile_report(title='Avacado Post-Profiling')   # Performing Post Profiling on data.
post_profile.to_file('post-profiling.html')                          # Saving report to post-profiling.html
# post_profile.to_notebook_iframe()                                    # View report inline here
data.head(2) # Preview of data
X_columns   # Preview of feature columns
X = data[X_columns]           # Features data

y = data[y_column]            # TV data
print(X.shape)

X.head()                      # Preview of X
print(y.shape)

y.head()                     # Preview of y
# Splitting the dataset into training and test sets 80-20 split.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# Reset index of split data sets

X_train.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)

y_test.reset_index(drop=True, inplace=True)
print(X_train.shape)

X_train.head()        # Preview of X_train
print(X_test.shape)

X_test.head()         # Preview of X_test
X_train_num = X_train[X_columns_num]       # Numeric X_train 

X_test_num = X_test[X_columns_num]         # Numeric X_test
from sklearn.preprocessing import StandardScaler         # Importing Standard Scalar

scaler = StandardScaler().fit(X_train_num)               # Fitting with train data
X_train_s = pd.DataFrame(scaler.transform(X_train_num), columns=X_columns_num)  # Transforming train data

X_test_s = pd.DataFrame(scaler.transform(X_test_num), columns=X_columns_num)    # Transforming test data
print(X_train_s.shape)

X_train_s.head()            # Scaled train data - Numeric
print(X_test_s.shape)

X_test_s.head()             # Scaled test data - Numeric
X_train[X_columns_cat].head()         # Preview of categorical features
# One Hot Encoding on Type for Train set.

X_train_type_dummies = pd.get_dummies(X_train['Type'], prefix='Type', drop_first=True)

print(X_train_type_dummies.shape)       # Shape of Dummies

X_train_type_dummies.head()             # Preview of Type Dummies               
X_train_s = pd.concat([X_train_s, X_train_type_dummies], 1) # Merging type dummies to Scaled Train set

print(X_train_s.shape)                                      # Shape of merged train set

X_train_s.head()                                            # Preview of merged train set
# One Hot Encoding on Type for Test set.

X_test_type_dummies = pd.get_dummies(X_test['Type'], prefix='Type', drop_first=True)

print(X_test_type_dummies.shape)       # Shape of Dummies

X_test_type_dummies.head()             # Preview of Type Dummies               
X_test_s = pd.concat([X_test_s, X_test_type_dummies], 1)   # Merging type dummies to Scaled test set

print(X_test_s.shape)                                      # Shape of merged test set

X_test_s.head()                                            # Preview of merged test set
# Label Encoding on Year for Train set.

from sklearn.preprocessing import LabelEncoder         # Importing Label Encoder

label_encoder = LabelEncoder().fit(X_train['Year'])    # Fitting on train set
X_train_year_dummies = pd.DataFrame(label_encoder.transform(X_train['Year']), columns=['Year'])

print(X_train_year_dummies.shape)       # Shape of Transformed Year

X_train_year_dummies.head()             # Preview of Transformed Year 
X_train_s = pd.concat([X_train_s, X_train_year_dummies], 1)   # Merging type dummies to Scaled train set

print(X_train_s.shape)                                        # Shape of merged train set

X_train_s.head()                                              # Preview of merged train set
X_test_year_dummies = pd.DataFrame(label_encoder.transform(X_test['Year']), columns=['Year'])

print(X_test_year_dummies.shape)       # Shape of Transformed Year

X_test_year_dummies.head()             # Preview of Transformed Year 
X_test_s = pd.concat([X_test_s, X_test_year_dummies], 1)   # Merging type dummies to Scaled test set

print(X_test_s.shape)                                      # Shape of merged test set

X_test_s.head()                                            # Preview of merged test set
# Installing category_encoders to import TargetEncoder

# !pip install category_encoders
# Label Encoding on Year for Train set.

from category_encoders import TargetEncoder                                # Importing Target Encoder

target_encoder_region = TargetEncoder().fit(X_train['Region'], y_train)    # Fitting on train set
X_train_region_dummies = target_encoder_region.transform(X_train['Region'])

print(X_train_region_dummies.shape)       # Shape of Transformed region

X_train_region_dummies.head()             # Preview of Transformed region 
X_train_s = pd.concat([X_train_s, X_train_region_dummies], 1)   # Merging region dummies to Scaled train set

print(X_train_s.shape)                                          # Shape of merged train set

X_train_s.head()                                                # Preview of merged train set
X_test_region_dummies = target_encoder_region.transform(X_test['Region'])

print(X_test_region_dummies.shape)       # Shape of Transformed region

X_test_region_dummies.head()             # Preview of Transformed region 
X_test_s = pd.concat([X_test_s, X_test_region_dummies], 1)     # Merging region dummies to Scaled train set

print(X_test_s.shape)                                          # Shape of merged train set

X_test_s.head()                                                # Preview of merged train set
target_encoder_month = TargetEncoder().fit(X_train['Month'], y_train)    # Fitting on train set for Month
X_train_month_dummies = target_encoder_month.transform(X_train['Month'])

print(X_train_month_dummies.shape)       # Shape of Transformed region

X_train_month_dummies.head()             # Preview of Transformed region 
X_train_s = pd.concat([X_train_s, X_train_month_dummies], 1)    # Merging region dummies to Scaled train set

print(X_train_s.shape)                                          # Shape of merged train set

X_train_s.head()                                                # Preview of merged train set
X_test_month_dummies = target_encoder_month.transform(X_test['Month'])

print(X_test_month_dummies.shape)       # Shape of Transformed region

X_test_month_dummies.head()             # Preview of Transformed region 
X_test_s = pd.concat([X_test_s, X_test_month_dummies], 1)      # Merging month dummies to Scaled train set

print(X_test_s.shape)                                          # Shape of merged train set

X_test_s.head()                                                # Preview of merged train set
print(X_train_s.shape)

X_train_s.head()                    # Preview of X_train
print(y_train.shape)

y_train.head()                    # Preview of y_train
print(X_test_s.shape)

X_test_s.head()                    # Preview of X_test
print(y_test.shape)

y_test.head()                    # Preview of y_test
# Importing Models

from sklearn.linear_model import LinearRegression              # Importing LinearRegression Algo

from sklearn.tree import DecisionTreeRegressor                 # Importing DecisionTreeRegressor Algo

from sklearn.ensemble import RandomForestRegressor             # Importing RandomForestRegressor Algo
# Creating our LinearRegression model and fitting the data into it.

linreg_model = LinearRegression()

linreg_model.fit(X_train_s, y_train)
# Creating our DecisionTreeRegressor model and fitting the data into it.

dt_model = DecisionTreeRegressor()

dt_model.fit(X_train_s, y_train)
# Creating our RandomForestRegressor model and fitting the data into it.

rf_model=RandomForestRegressor()

rf_model.fit(X_train_s,y_train)
# Preparations for Hyper Parameter Tuning



from sklearn.model_selection import GridSearchCV          # Importing GridSearchCV

from sklearn.model_selection import RandomizedSearchCV    # Importing RandomizedSearchCV



n_estimators = [10,50,100,200,300,500]                    # Number of trees in random forest

max_features = ['auto', 'log2',2,4,8,12]                  # Number of features to consider at every split

max_depth = [2,4,8,16,25]                                 # Maximum number of levels in tree=



# Creating param_grid for hyper-parameter tuning.

random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,}
# Creating our RandomForestRegressor model from GridSearchCV and fitting the data into it.

rf_model_grid = GridSearchCV(estimator = rf_model, param_grid=random_grid, cv = 3, n_jobs = -1 )

rf_model_grid.fit(X_train_s,y_train)
# Creating our RandomForestRegressor model from RandomizedSearchCV and fitting the data into it.

rf_model_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, 

                                     n_iter = 10, cv = 3, verbose=2, random_state=100, n_jobs = -1)

rf_model_random.fit(X_train_s, y_train)
# Predictions from `linreg_model` - TRAIN Set

y_train_pred_lr = linreg_model.predict(X_train_s)     # Predicted Target Values for TRAIN set.

print(y_train_pred_lr.shape)                          # Shape of Predicted Target Value - TRAIN set.

y_train_pred_lr[:10]                                  # Top 10 Predicted Target Values for TRAIN set.
# Predictions from `linreg_model` - TEST Set

y_test_pred_lr = linreg_model.predict(X_test_s)     # Predicted Target Values for TEST set.

print(y_test_pred_lr.shape)                         # Shape of Predicted Target Value - TEST set.

y_test_pred_lr[:10]                                 # Top 10 Predicted Target Values for TEST set.
# Predictions from `dt_model` - TRAIN Set

y_train_pred_dt = dt_model.predict(X_train_s)         # Predicted Target Values for TRAIN set.

print(y_train_pred_dt.shape)                          # Shape of Predicted Target Value - TRAIN set.

y_train_pred_dt[:10]                                  # Top 10 Predicted Target Values for TRAIN set.
# Predictions from `dt_model` - TEST Set

y_test_pred_dt = dt_model.predict(X_test_s)          # Predicted Target Values for TEST set.

print(y_test_pred_dt.shape)                          # Shape of Predicted Target Value - TEST set.

y_test_pred_dt[:10]                                  # Top 10 Predicted Target Values for TEST set.
# Predictions from `rf_model` - TRAIN Set

y_train_pred_rf = rf_model.predict(X_train_s)         # Predicted Target Values for TRAIN set.

print(y_train_pred_rf.shape)                          # Shape of Predicted Target Value - TRAIN set.

y_train_pred_rf[:10]                                  # Top 10 Predicted Target Values for TRAIN set.
# Predictions from `rf_model` - TEST Set

y_test_pred_rf = rf_model.predict(X_test_s)          # Predicted Target Values for TEST set.

print(y_test_pred_rf.shape)                          # Shape of Predicted Target Value - TEST set.

y_test_pred_rf[:10]                                  # Top 10 Predicted Target Values for TEST set.
# Predictions from `rf_model_grid` - TRAIN Set

y_train_pred_rf_grid = rf_model_grid.predict(X_train_s)    # Predicted Target Values for TRAIN set.

print(y_train_pred_rf_grid.shape)                          # Shape of Predicted Target Value - TRAIN set.

y_train_pred_rf_grid[:10]                                  # Top 10 Predicted Target Values for TRAIN set.
# Predictions from `rf_model_grid` - TEST Set

y_test_pred_rf_grid = rf_model_grid.predict(X_test_s)     # Predicted Target Values for TEST set.

print(y_test_pred_rf_grid.shape)                          # Shape of Predicted Target Value - TEST set.

y_test_pred_rf_grid[:10]                                  # Top 10 Predicted Target Values for TEST set.
# Predictions from `rf_model_random` - TRAIN Set

y_train_pred_rf_random = rf_model_random.predict(X_train_s)  # Predicted Target Values for TRAIN set.

print(y_train_pred_rf_random.shape)                          # Shape of Predicted Target Value - TRAIN set.

y_train_pred_rf_random[:10]                                  # Top 10 Predicted Target Values for TRAIN set.
# Predictions from `rf_model_random` - TEST Set

y_test_pred_rf_random = rf_model_random.predict(X_test_s)     # Predicted Target Values for TEST set.

print(y_test_pred_rf_random.shape)                          # Shape of Predicted Target Value - TEST set.

y_test_pred_rf_random[:10]                                  # Top 10 Predicted Target Values for TEST set.
# Utility / Helper Function - Regression Model Evaluation



def regression_model_evaluation(y, y_pred, set_type='', features_count=None):

    '''

    Utility/Helper method to calulate the Evaluation parameters for a regression model

    '''

    from sklearn import metrics # Importing metrics from SK-Learn

    result = {}

    

    if set_type != '':

        set_type = '_'+set_type

        

    # Mean Absolute Error on train set.

    result['MAE'] = metrics.mean_absolute_error(y, y_pred) 

    # Mean Squared Error on train set.

    result['MSE'] = metrics.mean_squared_error(y, y_pred)  

    # Root Mean Squared Error on train set.

    result['RMSE'] = np.sqrt(result['MSE'])                      

    # R_squared on train set.

    result['R_squared'] = metrics.r2_score(y, y_pred)      

    

    # Adj r2 = 1-(1-R2)*(n-1)/(n-p-1)

    if features_count:

        # Adjusted R_squared on train set.

        result['Adj_R_squared'] = 1 - (((1 - result['R_squared']) * (len(y)-features_count))/(len(y)-features_count-1))

    # Returning with appending type to key and rounding value 

    return {f'{k}'+set_type: round(v, 4) for k, v in result.items()} 
# Evaluation metrics for LinearRegression - TRAIN set

metrics_lr_train = regression_model_evaluation(y_train, y_train_pred_lr, features_count=8)

metrics_lr_train
# Evaluation metrics for LinearRegression - TEST set

metrics_lr_test = regression_model_evaluation(y_test, y_test_pred_lr, features_count=8)

metrics_lr_test
# Converting metrics map to DataFrame

LR_Train_mertrics = pd.DataFrame(metrics_lr_train.items(), columns=['Metrics', 'LR_Train'])

LR_Test_mertrics = pd.DataFrame(metrics_lr_test.items(), columns=['Metrics', 'LR_Test'])
# To get the intercept of the model.

linreg_model.intercept_
# To get the coefficients of the model.

coefs = linreg_model.coef_

features = X_train_s.columns



list(zip(features,coefs))
# Evaluation metrics for DecisionTreeRegressor - TRAIN set

metrics_dt_train = regression_model_evaluation(y_train, y_train_pred_dt, features_count=8)

metrics_dt_train
# Evaluation metrics for DecisionTreeRegressor - TEST set

metrics_dt_test = regression_model_evaluation(y_test, y_test_pred_dt, features_count=8)

metrics_dt_test
# Converting metrics map to DataFrame

DT_Train_mertrics = pd.DataFrame(metrics_dt_train.items(), columns=['Metrics', 'DT_Train'])

DT_Test_mertrics = pd.DataFrame(metrics_dt_test.items(), columns=['Metrics', 'DT_Test'])
# DecisionTreeRegressor Score; Same as R-Squared from X and y; So it internally calculates r-squared of y and y_pred (-from X)

print('Train set: ',dt_model.score(X_train_s,y_train))

print('Test set: ',dt_model.score(X_test_s,y_test))
# Evaluation metrics for RandomForestRegressor - TRAIN set

metrics_rf_train = regression_model_evaluation(y_train, y_train_pred_rf, features_count=8)

metrics_rf_train
# Evaluation metrics for RandomForestRegressor - TEST set

metrics_rf_test = regression_model_evaluation(y_test, y_test_pred_rf, features_count=8)

metrics_rf_test
# Converting metrics map to DataFrame

RF_Train_mertrics = pd.DataFrame(metrics_rf_train.items(), columns=['Metrics', 'RF_Train'])

RF_Test_mertrics = pd.DataFrame(metrics_rf_test.items(), columns=['Metrics', 'RF_Test'])
# RandomForestRegressor Score; Same as R-Squared from X and y; So it internally calculates r-squared of y and y_pred (-from X)

print('Train set: ',rf_model.score(X_train_s,y_train))

print('Test set: ',rf_model.score(X_test_s,y_test))
# Evaluation metrics for RandomForestRegressor with GridSearchCV - TRAIN set

metrics_rf_grid_train = regression_model_evaluation(y_train, y_train_pred_rf_grid, features_count=8)

metrics_rf_grid_train
# Evaluation metrics for RandomForestRegressor with GridSearchCV - TEST set

metrics_rf_grid_test = regression_model_evaluation(y_test, y_test_pred_rf_grid, features_count=8)

metrics_rf_grid_test
# Converting metrics map to DataFrame

RF_Grid_Train_mertrics = pd.DataFrame(metrics_rf_grid_train.items(), columns=['Metrics', 'RF_Grid_Train'])

RF_Grid_Test_mertrics = pd.DataFrame(metrics_rf_grid_test.items(), columns=['Metrics', 'RF_Grid_Test'])
# Evaluation metrics for RandomForestRegressor with RandomizedSearchCV - TRAIN set

metrics_rf_random_train = regression_model_evaluation(y_train, y_train_pred_rf_random, features_count=8)

metrics_rf_random_train
# Evaluation metrics for RandomForestRegressor with RandomizedSearchCV - TEST set

metrics_rf_random_test = regression_model_evaluation(y_test, y_test_pred_rf_random, features_count=8)

metrics_rf_random_test
# Converting metrics map to DataFrame

RF_Random_Train_mertrics = pd.DataFrame(metrics_rf_random_train.items(), columns=['Metrics', 'RF_Random_Train'])

RF_Random_Test_mertrics = pd.DataFrame(metrics_rf_random_test.items(), columns=['Metrics', 'RF_Random_Test'])
# Converting Train metrics df

Train_mertrics = LR_Train_mertrics.merge(

                    DT_Train_mertrics, on='Metrics').merge(

                    RF_Train_mertrics, on='Metrics').merge(

                    RF_Grid_Train_mertrics, on='Metrics').merge(

                    RF_Random_Train_mertrics, on='Metrics').set_index(keys='Metrics')

Train_mertrics
# Converting Train metrics df

Test_mertrics = LR_Test_mertrics.merge(DT_Test_mertrics, on='Metrics').merge(

                    RF_Test_mertrics, on='Metrics').merge(

                    RF_Grid_Test_mertrics, on='Metrics').merge(

                    RF_Random_Test_mertrics, on='Metrics').set_index(keys='Metrics')

Test_mertrics
model_mertrics = Train_mertrics.merge(Test_mertrics, on='Metrics')

model_mertrics = model_mertrics.reindex(

    columns=['LR_Train', 'LR_Test', 'DT_Train', 'DT_Test', 'RF_Train', 'RF_Test', 'RF_Grid_Train', 'RF_Grid_Test', 'RF_Random_Train', 'RF_Random_Test'])



model_mertrics
train_diff = pd.DataFrame({'Y_ACT':y_train , 'Y_Pred':y_train_pred_lr},columns=['Y_ACT','Y_Pred']) 

train_diff.head()    # Preview of DF - y_train and y_train_pred
test_diff = pd.DataFrame({'Y_ACT':y_test , 'Y_Pred':y_test_pred_lr},columns=['Y_ACT','Y_Pred'])

test_diff.head()    # Preview of DF - y_test and y_test_pred
f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))

f.suptitle('Y-Actual VS Y-Predicted - LinearRegression')

ax1.set_title('Train Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=train_diff, ax=ax1)

ax2.set_title('Test Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=test_diff, ax=ax2)
train_diff = pd.DataFrame({'Y_ACT':y_train , 'Y_Pred':y_train_pred_dt},columns=['Y_ACT','Y_Pred'])

train_diff.head() # Preview of DF - y_train and y_train_pred
test_diff = pd.DataFrame({'Y_ACT':y_test , 'Y_Pred':y_test_pred_dt},columns=['Y_ACT','Y_Pred'])

test_diff.head()   # Preview of DF - y_test and y_test_pred
f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))

f.suptitle('Y-Actual VS Y-Predicted - DecisionTreeRegressor')

ax1.set_title('Train Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=train_diff, ax=ax1)

ax2.set_title('Test Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=test_diff, ax=ax2)
train_diff = pd.DataFrame({'Y_ACT':y_train , 'Y_Pred':y_train_pred_rf},columns=['Y_ACT','Y_Pred'])

train_diff.head() # Preview of DF - y_train and y_train_pred
test_diff = pd.DataFrame({'Y_ACT':y_test , 'Y_Pred':y_test_pred_rf},columns=['Y_ACT','Y_Pred'])

test_diff.head()   # Preview of DF - y_test and y_test_pred
f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))

f.suptitle('Y-Actual VS Y-Predicted - RandomForestRegressor')

ax1.set_title('Train Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=train_diff, ax=ax1)

ax2.set_title('Test Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=test_diff, ax=ax2)
train_diff = pd.DataFrame({'Y_ACT':y_train , 'Y_Pred':y_train_pred_rf_grid},columns=['Y_ACT','Y_Pred'])

train_diff.head() # Preview of DF - y_train and y_train_pred
test_diff = pd.DataFrame({'Y_ACT':y_test , 'Y_Pred':y_test_pred_rf_grid},columns=['Y_ACT','Y_Pred'])

test_diff.head()  # Preview of DF - y_test and y_test_pred
f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))

f.suptitle('Y-Actual VS Y-Predicted - RandomForestRegressor With GridSearchCV')

ax1.set_title('Train Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=train_diff, ax=ax1)

ax2.set_title('Test Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=test_diff, ax=ax2)
train_diff = pd.DataFrame({'Y_ACT':y_train , 'Y_Pred':y_train_pred_rf_random},columns=['Y_ACT','Y_Pred'])

train_diff.head() # Preview of DF - y_train and y_train_pred
test_diff = pd.DataFrame({'Y_ACT':y_test , 'Y_Pred':y_test_pred_rf_random},columns=['Y_ACT','Y_Pred'])

test_diff.head() # Preview of DF - y_test and y_test_pred
f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))

f.suptitle('Y-Actual VS Y-Predicted - RandomForestRegressor With RandomizedSearchCV')

ax1.set_title('Train Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=train_diff, ax=ax1)

ax2.set_title('Test Set', fontsize=14)

sns.regplot(x='Y_ACT',y='Y_Pred',data=test_diff, ax=ax2)
model_mertrics # Preview of Model Evaluation Metrics