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
# graphs
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings

# stats
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from collections import Counter

# models
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn import svm
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression,LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import confusion_matrix # creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import precision_score, accuracy_score

# pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')

%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline


# Loading the data set
apt_price=pd.read_csv("../input/seoul-apt-prices-20062012/APT_price_seoul_2006_2012.csv")

#change the unit of price from Won to Dollar
apt_price["price"] = apt_price["price"]*10

# Set of features to be used 
features_selected = ['District', 'maxBuild', 'Hhld', 'Floor', 'Size', 'schoolDistHs', 'new..prop..snu23','Age_complex', 'yearmon', 'BuildId']

def select_dat_specific_year(yr, fraction=None):
    # Select data of the APTs that were sold in yr
    if fraction is None:
        apt_price12 = apt_price[apt_price['year'] == yr]
    else:
        apt_price12 = apt_price[apt_price['year'] == yr]
        apt_price12 = apt_price12.sample(frac=fraction)

    # Remove rows with missing target, separate target from predictors
    X_full = apt_price12.dropna(axis=0, subset=['price'])

    X = X_full[features_selected]
    y = X_full.price
    
    return X, y


# Making train/validation data set

X, y = select_dat_specific_year(2012)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)


# Select categorical columns with dtype=object that will be encoded in a later part
categorical_cols0 = [cname for cname in X_train_full.columns if 
                    X_train_full[cname].dtype == "object"]
categorical_cols = list(set(categorical_cols0) - set(['yearmon', 'BuildId']))

# Select numerical columns that will be imputed in a later part
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]


# Keep selected columns only
my_cols = categorical_cols0 + numerical_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Making test data set (we will use 2011 data as test data)

X_test_full, y_test_full = select_dat_specific_year(2011, fraction=0.1)
X_test = X_test_full[my_cols].copy()
y_test = y_test_full.copy()
# Verify the number of missing values for each numeric variable
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Fill in the lines below: imputation

# Imputation for numerical_cols
my_imputer = SimpleImputer(strategy="mean") 
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train[numerical_cols]))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid[numerical_cols]))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test[numerical_cols]))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = numerical_cols
imputed_X_valid.columns = numerical_cols
imputed_X_test.columns = numerical_cols

# Put index
imputed_X_train.index=X_train.index
imputed_X_valid.index=X_valid.index
imputed_X_test.index=X_test.index

# condition
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# do one-hot encode the categorical_cols
OH_X_train0 = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_cols])) 
OH_X_valid0 = pd.DataFrame(OH_encoder.transform(X_valid[categorical_cols]))
OH_X_test0 = pd.DataFrame(OH_encoder.transform(X_test[categorical_cols]))

# put index 
OH_X_train0.index = X_train.index
OH_X_valid0.index = X_valid.index
OH_X_test0.index = X_test.index

# combine encoded categorical_cols with numeric variables + rest of variables ('BuildId', 'yearmon')
OH_X_train = pd.concat([imputed_X_train, OH_X_train0, X_train[['BuildId', 'yearmon']]], axis=1)
OH_X_valid = pd.concat([imputed_X_valid, OH_X_valid0, X_valid[['BuildId', 'yearmon']]], axis=1)
OH_X_test = pd.concat([imputed_X_test, OH_X_test0, X_test[['BuildId', 'yearmon']]], axis=1)
# Add the second power of size to capture the nonlinearity of the effect of size on house prices.

def second_power(series):
    # series is changed to numpy array.
    
    tmp=np.array(series)
    tmp2=tmp**2
    
    # the new series is changed to panda series.
    
    tmp3=pd.Series(tmp2)
    tmp3.index=series.index
    
    return tmp3

# Apply the fct defined above.

OH_X_train['Size2']=second_power(OH_X_train['Size'])
OH_X_valid['Size2']=second_power(OH_X_valid['Size'])
OH_X_test['Size2']=second_power(OH_X_test['Size'])


# Verify if there is missing values in the new series

OH_X_train['Size2'].isnull().sum()
OH_X_valid['Size2'].isnull().sum()
OH_X_test['Size2'].isnull().sum()
# Add the number of APT sales of the month happened by APT complex

def count_past_sales(series):
    
    # time_stamp
    
    series2 = pd.to_datetime(series)
    sale_time = pd.Series(series2.index, index=series2, name='count_sales_this_month').sort_index() # exchange the positions of index and values
    count_1month= sale_time.rolling('30D', min_periods=1).count()
    count_1month_2=count_1month.groupby(count_1month.index.month).transform('last')
    return count_1month_2

aa=OH_X_train.groupby('BuildId')['yearmon'].apply(count_past_sales)
bb=OH_X_valid.groupby('BuildId')['yearmon'].apply(count_past_sales)
cc=OH_X_test.groupby('BuildId')['yearmon'].apply(count_past_sales)

# put index
aa.index=OH_X_train.sort_values(by=['BuildId','yearmon']).index
bb.index=OH_X_valid.sort_values(by=['BuildId','yearmon']).index
cc.index=OH_X_test.sort_values(by=['BuildId','yearmon']).index

OH_X_train['num_sales_same_month_by_complex']=aa
OH_X_valid['num_sales_same_month_by_complex']=bb
OH_X_test['num_sales_same_month_by_complex']=cc

# drop 'BuildId' and 'yearmon' columns
OH_X_train.drop(['BuildId', 'yearmon'], axis=1, inplace=True)
OH_X_valid.drop(['BuildId', 'yearmon'], axis=1, inplace=True)
OH_X_test.drop(['BuildId', 'yearmon'], axis=1, inplace=True)

# Select the optimal set of features (using LogisticRegression classifier)
## We search for the best value of 'K' which is the number of variables that will be used in training

# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectFromModel

# def select_features_l1(X, y):
#     """Return selected features using logistic regression with an L1 penalty."""
#     logistic=LogisticRegression(C=0.1, penalty="l1", solver='liblinear', random_state=7).fit(X,y)
#     model=SelectFromModel(logistic, prefit=True)

#     X_new=model.transform(X)
#     selected_features=pd.DataFrame(model.inverse_transform(X_new), index=X.index, columns=X.columns)
#     selected_col=selected_features.columns[selected_features.var()!=0]
#     return selected_col

# select_col=select_features_l1(OH_X_train, y_train)

# X_train_reg=OH_X_train[select_col]
# X_valid_reg=OH_X_valid[select_col]
# X_test_reg=OH_X_test[select_col]
# Description of the APT prices (target variable)
y_train.describe()
# Plot Histogram
sns.distplot(y_train, fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(y_train)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(y_train, plot=plt)
plt.show()

print("Skewness: %f" % y_train.skew())
print("Kurtosis: %f" % y_train.kurt())
# We do a same analysis using log(price)
logy = np.log(y_train)

# Plot Histogram
sns.distplot(logy, fit=norm)

# Getting the fitted parameters used by the function
(mu, sigma) = norm.fit(logy)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(logy, plot=plt)
plt.show()

print("Skewness: %f" % logy.skew())
print("Kurtosis: %f" % logy.kurt())
# APT house prices by district

var = 'District'
data = pd.concat([y_train, X_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="price", data=data)
fig.set_xticklabels(ax.get_xticklabels(),rotation=90)
fig.axis(ymin=0, ymax=3000000)

# APT house prices by school district

var = 'schoolDistHs'
data = pd.concat([y_train, X_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6));
fig = sns.boxplot(x=var, y="price", data=data)
fig.set_xticklabels(ax.get_xticklabels(),rotation=30)
fig.set_xlabel('School District')
fig.set_ylabel('APT Price')
fig.axis(ymin=0, ymax=3000000)

# School quality by school district

var = 'schoolDistHs'
xx = X_train['new..prop..snu23']
data = pd.concat([xx, X_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6));
fig = sns.boxplot(x='schoolDistHs', y='new..prop..snu23', data=data)
fig.set_xticklabels(ax.get_xticklabels(),rotation=30)
fig.set_xlabel('School District')
fig.set_ylabel('School quality')
fig.axis(ymin=0, ymax=5)
# Apt size vs Sale Price

fig= sns.jointplot(x=X_train['Size'], y= y_train, kind="reg" )
fig.ax_joint.set_xlabel('APT size')
fig.ax_joint.set_ylabel('Price')
# Apt floor vs Sale Price


fig = sns.jointplot(x=X_train['Floor'], y= y_train, kind='reg')
fig.ax_joint.set_xlabel('Floor')
fig.ax_joint.set_ylabel('Price')
# School quality vs Sale Price

fig = sns.jointplot(x=X_train['new..prop..snu23'], y= y_train, kind='reg')
fig.ax_joint.set_xlabel('School quality')
fig.ax_joint.set_ylabel('Price')
# Number of Households vs Sale Price

fig = sns.jointplot(x=X_train['Hhld'], y= y_train, kind='reg')
fig.ax_joint.set_xlabel('Number of households')
fig.ax_joint.set_ylabel('Price')
# No building vs Sale Price

fig = sns.jointplot(x=X_train['maxBuild'], y= y_train, kind='reg')
fig.ax_joint.set_xlabel('Number of building')
fig.ax_joint.set_ylabel('Price')
# Define a fct to get the mean absolute error values for each 'min_samples_leaf' values

def get_mae(msl, train_X, val_X, train_y, val_y):
    
    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(n_estimators=400, random_state=1, criterion = "mse", min_samples_leaf=msl, max_features=20)

    # Fit your model
    rf_model.fit(train_X, train_y)
    
    # predict
    pred = rf_model.predict(val_X)
    
    # Calculate the mean absolute error of your Random Forest model on the validation data
    rf_val_mae = mean_absolute_error(val_y, pred)

    print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
   
    return(rf_val_mae)

# Repeated training with different set of parameters 'min_samples_leaf'

candidate_min_samples_leaf = [1,2,3,5,10,20,40,80]

# Write a loop function to find the ideal tree size from candidate_max_leaf_nodes

set_mae=[] 

for msl in candidate_min_samples_leaf:
    set_mae.append(get_mae(msl, OH_X_train, OH_X_valid, y_train, y_valid))
    

# Store the best value of min_samples_leaf 
best_leaf_size = candidate_min_samples_leaf[set_mae.index(min(set_mae))]    

print("Value of MAE for the optimal Random Forest Model is as follows:")

get_mae(best_leaf_size, OH_X_train, OH_X_valid, y_train, y_valid)


print("Value of MAE for the optimal Random Forest Model is as follows (test data):")

get_mae(best_leaf_size, OH_X_train, OH_X_test, y_train, y_test)

print("The best parameter for 'min_samples_leaf' is:", best_leaf_size)

# Define a fct to get mae values. We use this fct to find an optimal set of parameters in using the XGboost algorithm

def get_mae_xgb(mymodel, earlyval=None, learning_rate=None, xdat = OH_X_valid, ydat= y_valid):
    
    my_model=mymodel
    
    if earlyval is not None:
        my_model.fit(OH_X_train, y_train, 
                     early_stopping_rounds=earlyval, 
                     eval_set=[(OH_X_valid, y_valid)],
                     verbose=False)  
    else:    
        my_model.fit(OH_X_train, y_train)
    
    predictions = my_model.predict(xdat)
    
    mae= mean_absolute_error(predictions,ydat)
    
    if earlyval is not None:
        print('===ealry stopping rounds vary===')
        print("Mean Absolute Error: " + str(mae))
    
    elif learning_rate is not None:
        print('===learning rates vary===')
        print("Mean Absolute Error: " + str(mae))
    
    else:
        print("Mean Absolute Error: " + str(mae))

    return mae

# Only with 'default parameter' values

my_model = XGBRegressor(random_state=0)


print("Value of MAE for the optimal Random Forest Model is as follows:")
get_mae_xgb(my_model)

print("Value of MAE for the optimal Random Forest Model is as follows (test data):")
get_mae_xgb(my_model,earlyval=None, learning_rate=None, xdat= OH_X_test, ydat= y_test)

# Varying 'n_estimators' values

nestim=[300,400,500,1000,1500,2000]

set_mae=[]

for nn in nestim:
    
    my_model1 = XGBRegressor(random_state=0, n_estimators=nn)
    set_mae.append(get_mae_xgb(my_model1))

    

# Mean absolute error values with the best parameter

best_nestime_size = nestim[set_mae.index(min(set_mae))]    
my_best_model1 = XGBRegressor(random_state=0, n_estimators=best_nestime_size)

print("Value of MAE for the optimal Random Forest Model is as follows:")
get_mae_xgb(my_best_model1)


print("Value of MAE for the optimal Random Forest Model is as follows (test data):")
get_mae_xgb(my_best_model1, xdat= OH_X_test, ydat= y_test)
print("The best parameter for 'n_estimator' is:", best_nestime_size)

# Varying 'early stopping rounds' values


vall=[5,10,15,20,25,30,40,50,100] # candidates of early stop rounds

set_mae=[]

for val in vall:
    my_model2 = XGBRegressor(random_state=0, n_estimators=best_nestime_size)
    set_mae.append(get_mae_xgb(my_model2, earlyval=val))

# Mean absolute error values with the best parameter

best_est = vall[set_mae.index(min(set_mae))]    

print("Value of MAE for the optimal Random Forest Model is as follows:")
get_mae_xgb(my_model2, earlyval=best_est)


print("Value of MAE for the optimal Random Forest Model is as follows (test data):")
get_mae_xgb(my_model2, earlyval=best_est, xdat= OH_X_test, ydat= y_test)
print("The best parameter for 'early stopping rounds' is:", best_est)

# Varying 'learning rate' values

LRval=[0.01,0.02,0.04, 0.05, 0.1, 0.2,0.4, 0.5]
val=best_est

set_mae=[]

for LRvall in LRval:
    my_model3 = XGBRegressor(random_state=0,n_estimators= best_nestime_size, learning_rate=LRvall)
    set_mae.append(get_mae_xgb(my_model3, earlyval=val, learning_rate = LRvall))
    
# Mean absolute error values with the best parameter

best_lr = LRval[set_mae.index(min(set_mae))]    

print("Value of MAE for the optimal Random Forest Model is as follows:")
get_mae_xgb(my_model3, earlyval=val, learning_rate=best_lr)


print("Value of MAE for the optimal Random Forest Model is as follows (test data):")
get_mae_xgb(my_model3, earlyval=val, learning_rate=best_lr, xdat= OH_X_test, ydat= y_test)
print("The best parameter for 'learning rate' is:", best_lr)

# Using GridSearchCV() to find the best combination of parameters

## First trial

# param_grid= {
#     'max_depth': [3,4,5],
#     'learning_rate': [0.1,0.01,0.05],
#     'gamma': [0,0.25,1],
#     'n_estimators': [500, 600, 700],   
# }

## 1st trial
#{'early_stopping_rounds': 30, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 700}


## Second trial 
# param_grid= {
#     'max_depth': [5,6,7],
#     'learning_rate': [0.1,0.2,0.3],
#     'gamma': [0,0.05,0.01],
#     'n_estimators': [700,800,900],
# }



## 2nd trial
#{'early_stopping_rounds': 30, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 900}

## Third trial 
# param_grid= {
#     'max_depth': [6],
#     'learning_rate': [0.1],
#     'gamma': [0],
#     'n_estimators': [1100,1500,2000,2500],
# }

## 3rd trial
#{'early_stopping_rounds': 30, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 1500} # best combination of parameters!!


## Best parameters obatined through GridSearchCV()

best_est = 30

param_grid = {
            'max_depth': [6],
            'learning_rate': [0.1],
            'gamma': [0],
            'n_estimators': [1500]
}

    
XGB=XGBRegressor(random_state=0, 
                         colsample_bytree=0.5)

XGB_grids = GridSearchCV(estimator = XGB,
                         param_grid=param_grid,
                         verbose=0,
                         cv=5)


# fitting the model for grid search 

XGB_grids.fit(OH_X_train, y_train,
              early_stopping_rounds = best_est,
              eval_set=[(OH_X_valid, y_valid)], 
             verbose=False)


# Results with the best parameters
print('The best model is: ', XGB_grids.best_params_)
print('This model produces a mean cross-validated score (precision) of', XGB_grids.best_score_)

# getting the mean absolute error using the test dataset and the best parameters
y_true, y_pred = y_valid, XGB_grids.predict(OH_X_valid)

print('MAE on the evaluation set: ',mean_absolute_error(y_true, y_pred))


# getting the mean absolute error using the test dataset and the best parameters
y_true, y_pred = y_test, XGB_grids.predict(OH_X_test)

print('MAE on the evaluation set: ',mean_absolute_error(y_true, y_pred))

# Modeling and finding the set of parameters optimal for the algorithm

modelsvr = SVR()

# first trial 
# param = {'kernel' : ['rbf'],
#          'C' : [1,3,5,7,9,11,13,15],
#          'gamma' : ['auto']}

# the best set of parameters

param = {'kernel' : ['rbf'],
         'C' : [9],
         'gamma' : ['auto']}

svr_grids = GridSearchCV(estimator= modelsvr,
                         param_grid= param,
                         cv=5)

svr_grids.fit(OH_X_train, y_train)



# Results with the best parameters

print(svr_grids.cv_results_)
print('The best model is: ', svr_grids.best_params_)
print('This model produces a mean cross-validated score (precision) of', svr_grids.best_score_)


# getting the mean absolute error using the test dataset and the best parameters

y_true, y_pred = y_test, svr_grids.predict(OH_X_test)
print('MAE on the evaluation set: ',mean_absolute_error(y_true, y_pred))



# Modeling and finding the set of parameters optimal for the algorithm

def sklearn_reg(train_data, label_data, test_data, k_num):
    knn = neighbors.KNeighborsRegressor(n_neighbors=k_num, weights='uniform', algorithm='auto')
    # Train
    knn.fit(train_data, label_data)
    # Predict
    predict_label = knn.predict(test_data)
    # Return
    return predict_label
                               
                               
# Predict using the training data set, assuming k = 5 

y_predict = sklearn_reg(OH_X_train, y_train, OH_X_valid, 5)
y_predict
                               

# mean absolute error calculation

def get_mae_knear(predictions, y_valid):  
    mae = mean_absolute_error(predictions, y_valid)
    return mae
    
                               
get_mae_knear(y_predict, y_valid)


# Search for the value 'k' that leads to the minimum absolute error values
                               
normal_mae = []  # Create an empty list of accuracy rates
k_value = range(2, 30)


for k in k_value:
    y_predict = sklearn_reg(OH_X_train, y_train, OH_X_valid, k)
    mae = get_mae_knear(y_valid, y_predict)
    normal_mae.append(mae)

# draw a graph of accuracy level across 'k'

plt.xlabel("k")
plt.ylabel("Mean absolute error")
new_ticks = np.linspace(0.6, 0.9, 30)  # Set the y-axis display
plt.yticks(new_ticks)
plt.plot(k_value, normal_mae, c='r')
plt.grid(True)  # Add grid
      

# calculate mae usig test data and best parameter 'k'
best_k=k_value[normal_mae.index(min(normal_mae))]
get_mae_knear(sklearn_reg(OH_X_train, y_train, OH_X_valid,best_k), y_valid)
get_mae_knear(sklearn_reg(OH_X_train, y_train, OH_X_test,best_k), y_test)

# Modeling and finding the set of parameters optimal for the algorithm

MLP = MLPClassifier(random_state=0)

# First trial
# param_list = {"hidden_layer_sizes": [(100,),(300,),(500,)], 
#               "activation": ["logistic", "relu"], 
#               "solver": ["lbfgs", "adam"], 
#               "alpha": [0.001,0.005], # L2 penalty (regularization term) parameter
#               "batch_size": [100,200]
#              }

# Second trial
# param_list = {"hidden_layer_sizes": [(100,),(50,50), (50,25,25)], 
#               "activation": ["logistic", "relu"], 
#               "solver": ["lbfgs", "adam"], 
#               "alpha": [0.001], # L2 penalty (regularization term) parameter
#               "batch_size": [100]
#              }

# Third trial

# param_list = {"hidden_layer_sizes": [(100,)], 
#               "activation": ["logistic", "relu"], 
#               "solver": ["lbfgs", "adam"], 
#               "alpha": [0.001,0.005], # L2 penalty (regularization term) parameter
#               "batch_size": [100,200]
#              }

# Final model (w the set of best parameters)

param_list = {"hidden_layer_sizes": [(100,)], 
              "activation": ["logistic"], 
              "solver": ["lbfgs"], 
              "alpha": [0.001], # L2 penalty (regularization term) parameter
              "batch_size": [100]
             }


MLP_grids = GridSearchCV(estimator= MLP,
                     param_grid= param_list,
                     cv=5)

MLP_grids.fit(OH_X_train, y_train)

# Results with the best parameters
print(MLP_grids.cv_results_)            
print('The best model is: ', MLP_grids.best_params_)
print('This model produces a mean cross-validated score (precision) of', MLP_grids.best_score_)


# getting the mean absolute error using the test dataset and the best parameters
y_true, y_pred = y_test, MLP_grids.predict(OH_X_test)
print('MAE on the evaluation set: ',mean_absolute_error(y_true, y_pred))
### Summmary of MAE values from different algorithms 

## Random Forest model 

print("Value of MAE for the Random Forest Model on the evaluation set is as follows (validation data):")
get_mae(best_leaf_size, OH_X_train, OH_X_valid, y_train, y_valid)

print("Value of MAE for the Random Forest Model on the evaluation set is as follows (test data):")
get_mae(best_leaf_size, OH_X_train, OH_X_test, y_train, y_test)


## XGBoost algorithm + Random forest model

y_true, y_pred = y_valid, XGB_grids.predict(OH_X_valid)
print('Value of MAE for the XGB on the evaluation set is as follows (validation data): ',mean_absolute_error(y_true, y_pred))

y_true, y_pred = y_test, XGB_grids.predict(OH_X_test)
print('Value of MAE for the XGB on the evaluation set is as follows (test data): ',mean_absolute_error(y_true, y_pred))


## support vector regressor
y_true, y_pred = y_valid, svr_grids.predict(OH_X_valid)
print('Value of MAE for the Support Vector regressor on the evaluation set (validation data): ',mean_absolute_error(y_true, y_pred))

y_true, y_pred = y_test, svr_grids.predict(OH_X_test)
print('Value of MAE for the Support Vector regressor on the evaluation set (test data): ',mean_absolute_error(y_true, y_pred))


##  k nearest algorithm 
best_k=k_value[normal_mae.index(min(normal_mae))]
print('Value of MAE for the k neareast algorithm on the evaluation set (validation data): ',get_mae_knear(sklearn_reg(OH_X_train, y_train, OH_X_valid,best_k), y_valid))
print('Value of MAE for the k neareast algorithm on the evaluation set (validation data): ',get_mae_knear(sklearn_reg(OH_X_train, y_train, OH_X_test,best_k), y_test))


##  Multi layer perceptron
y_true, y_pred = y_valid, MLP_grids.predict(OH_X_valid)
print('Value of MAE for the Multi layer perceptron on the evaluation set (validation data): ',mean_absolute_error(y_true, y_pred))
y_true, y_pred = y_test, MLP_grids.predict(OH_X_test)
print('Value of MAE on  the Multi layer perceptron on the evaluation set (test data): ',mean_absolute_error(y_true, y_pred))


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data using "Pipeline"
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Bundle preprocessing for numerical and categorical data using "ColumnTransformer"
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])



# Define model (optmal model + optimal parameter)
model = RandomForestRegressor(n_estimators=100, random_state=0)


# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)


# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))


# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)


# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

