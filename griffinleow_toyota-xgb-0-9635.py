# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import time







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Toyota dataset

# We want to look at how the data looks like before generalising with other datasets

allToyota = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/toyota.csv')



# Check out head()

print(allToyota.head())



# Check shape of dataset

print('Shape of dataset: ', allToyota.shape)

print('\n')



# Check for basic info about dataset i.e. any missing or null values

print(allToyota.info())
# use str.strip()

features = ['model', 'transmission', 'fuelType']

for feature in features:

    allToyota[feature] = allToyota[feature].apply(lambda x: x.strip())
sns.set_style("whitegrid")



fig, axes = plt.subplots(2,2, figsize=(22,15))

features = ['price','mileage','tax','year']

for i, ax in zip(range(len(features)), axes.flat):

    sns.distplot(allToyota[features[i]], bins=20, kde=False, ax=ax)

    plt.title('Distribution of ' + features[i])

    plt.xlabel(str(features[i]))

plt.subplots(figsize=(20,10))

# Price is our dependent variable; year is our independent variable

sns.scatterplot(x = 'year', y = 'price', hue = 'model', data = allToyota, s = 100, alpha = 0.7)

plt.title('Price of cars vs year')

plt.xlabel('Year')

plt.ylabel('Price of Car')

plt.show()
plt.subplots(figsize=(20,10))

# Price is our dependent variable; year is our independent variable

sns.boxplot(x = 'year', y = 'price', data = allToyota)

plt.title('Price of cars vs year')

plt.xlabel('Year')

plt.ylabel('Price of Car')

plt.show()
plt.subplots(figsize=(20,10))

# Price is our dependent variable; year is our independent variable

sns.scatterplot(x = 'mileage', y = 'price', hue = 'model', data = allToyota, s = 100, alpha = 0.7)

plt.title('Price of cars vs year')

plt.xlabel('Mileage')

plt.ylabel('Price of Car')

plt.show()
plt.subplots(figsize=(20,10))

# Price is our dependent variable; year is our independent variable

sns.boxplot(x = 'model', y = 'price', data = allToyota)

plt.title('Variation of prices for each model')

plt.xlabel('Model')

plt.ylabel('Price of Car')

plt.show()
supraAndIQ = allToyota.loc[allToyota['model'].isin(['Supra', 'IQ'])]

print('Shape of supraAndIQ is:', supraAndIQ.shape)



plt.subplots(figsize=(13,6))

# Price is our dependent variable; year is our independent variable

sns.scatterplot(x = 'year', y = 'price', hue = 'model', data = supraAndIQ, s = 100, alpha = 0.7)

plt.title('Price vs Year')

plt.xlabel('Registration Year')

plt.ylabel('Price')

plt.show()
plt.subplots(figsize=(13,6))

sns.scatterplot(x = 'mileage', y = 'price', hue = 'transmission', data = supraAndIQ[supraAndIQ['model'] == 'Supra'], s = 100, alpha = 0.7)

plt.title('Price vs Mileage')

plt.xlabel('Mileage')

plt.ylabel('Price')

plt.show()
plt.subplots(figsize=(13,6))

sns.scatterplot(x = 'mileage', y = 'price', hue = 'transmission', data = supraAndIQ[supraAndIQ['model'] == 'IQ'], s = 100, alpha = 0.7)

plt.title('Price vs Mileage')

plt.xlabel('Mileage')

plt.ylabel('Price')

plt.show()
plt.subplots(figsize=(13,6))

sns.scatterplot(x = 'mileage', y = 'price', hue = 'transmission', data = allToyota[allToyota['model'] == 'Land Cruiser'], s = 100, alpha = 0.7)

plt.title('Price vs Mileage')

plt.xlabel('Mileage')

plt.ylabel('Price')

plt.show()
# Need to groupby to know how many cars per model

allToyota.head()



# modelGroupby = allToyota.groupby(['model'])['price'].count().reset_index()

modelGroupby = allToyota.groupby(['model'])['price'].count().reset_index().sort_values(['price'], ascending=False)



modelGroupby = modelGroupby.rename(columns={'price':'count'})



plt.subplots(figsize=(20,10))

# Price is our dependent variable; year is our independent variable

sns.barplot(x = 'model', y = 'count', data = modelGroupby)

plt.title('Sale for Each Model')

plt.xlabel('Model')

plt.ylabel('No. of Cars Sold')

plt.show()
# What are the categorical columns?

categorical_cols = allToyota.select_dtypes('object').columns

print(categorical_cols)



def encode_categoricals(df, cols):

    # df - input dataframe

    # cols - categorical column names

    

    # Steps to dummy encode

    # 1. Create dummy_encode dataframe

    # 2. Concat to main dataframe

    # 3. Drop original columns

    

    for col in cols:

        df_dummy = pd.get_dummies(df[col], prefix = col)

        df = pd.concat([df, df_dummy], axis = 1)

        df.drop([col], axis = 1, inplace = True)

        

    return df



allToyotaEncoded = encode_categoricals(allToyota, categorical_cols)

allToyotaEncoded.head()
from sklearn.model_selection import train_test_split



label = allToyotaEncoded['price']

features = allToyotaEncoded.drop(columns=['price'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.1, random_state = 123)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, BayesianRidge, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn import metrics

from xgboost import XGBRegressor

import lightgbm as lgb



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# Define a generic model to calculate the R2_score between the true and predicted value

def kfold_cv_scoring(model, scoring, folds, x_train, y_train):

    kf_cv = KFold(folds, shuffle=True, random_state = 123).get_n_splits(x_train.values)

#     score = np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring = scoring, cv = kf_cv))

    score = cross_val_score(model, x_train.values, y_train, scoring = scoring, cv = kf_cv)



    return score



# Define a function to plot the results of the k-fold cross validation training:

def plot_kfold_cv_results(model, folds, x_train, y_train, model_name, scoring, result_table):

    # model - model class

    # model_name - name of model

    # result_time - to hold the summary of results

    

    start = time.time()

    model_scores = kfold_cv_scoring(model, scoring, folds, x_train, y_train)

    

    end = time.time()

    duration = (end-start)/60

    print('{} took {:.3f} minutes to complete training'.format(model_name, duration))

    

    # Plot the graph of RMSE for different folds

    plt.plot(model_scores)

    plt.xlabel('ith interation of Kfolds')

    plt.ylabel(str(scoring))

    plt.title(str(scoring) + ' for different iteration of K-folds')

    plt.show()

    

    # Print out the mean and standard deviation of the scoring values

    scoring_mean = model_scores.mean()

    scoring_std = model_scores.std()

    print('Mean ' + str(scoring) + ' is {:.4f}'.format(scoring_mean))

    print('Std ' + str(scoring) + ' is {:.4f}'.format(scoring_std))

    print('\n')

    

    # Append results to result_table

    new_row = [model_name, scoring_mean, scoring_std]

    result_table.loc[len(result_table)] = new_row

#     result_table.sort_values(by = [str(scoring)])

    print(result_table)

    

    return None
### Initialize the 'Result_Table' first

result_table = pd.DataFrame(columns = ['Model', 'Mean', 'Std'])



# Define all the models here

lm = LinearRegression()

kernelridge = KernelRidge(alpha = 0.6, kernel='polynomial', degree=2, coef0=2.5)

gradboost = GradientBoostingRegressor(

    n_estimators=500,

    learning_rate=0.001,

    max_depth=4,

    max_features='sqrt',

    loss='huber',

    random_state=123,

    criterion='friedman_mse'

)

elasticnet = make_pipeline(RobustScaler(), ElasticNet(max_iter=1e7, alpha=0.0005, l1_ratio=0.9, random_state=123))

xgb = XGBRegressor(

    colsample_bytree=0.2,

    gamma=0.0,

    learning_rate=0.01,

    max_depth=5, 

    min_child_weight=1.5,

    n_estimators=4000,

    reg_alpha=0.9,

    reg_lambda=0.6,

    subsample=0.8,

    verbosity=0,

    random_state = 7,

    objective='reg:squarederror',

    n_jobs = -1

)

lightGBM = lgb.LGBMRegressor(

    objective='regression',

    num_leaves=10,

    learning_rate=0.05,

    n_estimators=1000,

    max_bin = 55,

    bagging_fraction = 0.8,

    bagging_freq = 5,

    feature_fraction = 0.2319,

    feature_fraction_seed=9,

    bagging_seed=9,

    min_data_in_leaf =10,

    min_sum_hessian_in_leaf = 11)

rf = RandomForestRegressor(n_estimators=500, max_depth=2, random_state=123)



models = [lm, kernelridge, gradboost, elasticnet, xgb, lightGBM, rf]

# Use for-loop to train all models defined above:

for model in models:

    print('Begin training for ', model)

    plot_kfold_cv_results(

        model=model,

        folds=5,

        x_train=X_train,

        y_train=y_train,

        model_name=str(model),

        scoring='r2',

        result_table=result_table)
# In summary, these are the baseline results

result_table.sort_values(by='Mean', ascending=False)
from sklearn.feature_selection import SelectFromModel

from xgboost import plot_importance

from numpy import sort



# Define our XGBRegressor model again

xgb = XGBRegressor(

    colsample_bytree=0.2,

    gamma=0.0,

    learning_rate=0.01,

    max_depth=5, 

    min_child_weight=1.5,

    n_estimators=4000,

    reg_alpha=0.9,

    reg_lambda=0.6,

    subsample=0.8,

    verbosity=0,

    random_state = 7,

    objective='reg:squarederror',

    n_jobs = -1

)



# Fit XGBR on all training data - X_train

xgb.fit(X_train, y_train)



# Plot features ranked according to their importances

fig, ax = plt.subplots(figsize=(20,15))

plot_importance(xgb, max_num_features=50, height=0.8, ax=ax)

plt.show()
print('Current number of features in X_train:', len(X_train.columns))
# Fit XGBR using each importance as a threshold

thresholds = sort(xgb.feature_importances_)

print('thresholds:', thresholds)



# Do a quick plot to see the feature importance

plt.subplots(figsize=(11, 8))

plt.scatter(x = range(0, len(thresholds)), y = thresholds, s=10)

plt.xlabel('n-th feature')

plt.ylabel('Feature_Importance')

plt.show()
# Store the threshold, no. of features, and corresponding R2 for purpose of visualisation

r2_feat_importance = pd.DataFrame(columns = ['thresholds', 'no_features', 'threshold_r2'])



# Define function to calculate R2:

def model_r2(y, y_pred):

    return r2_score(y, y_pred)



start = time.time()



# For thresh values in interval of 2 units (depends on how many features and how long you want to spend iterating on)

for i in range(0, len(thresholds)):

    if i % 2 == 0: # multiples of 2

        print('Current index is:', i)

        

        thresh = thresholds[i]

        # For thresh values in interval of i units:

        # select features using threshold

        selection = SelectFromModel(xgb, threshold = thresh, prefit = True)

        select_X_train = selection.transform(X_train)

            

        # Define model again

        selection_model = XGBRegressor(

            colsample_bytree=0.2,

            gamma=0.0,

            learning_rate=0.01,

            max_depth=5, 

            min_child_weight=1.5,

            n_estimators=4000,

            reg_alpha=0.9,

            reg_lambda=0.6,

            subsample=0.8,

            verbosity=0,

            random_state = 7,

            objective='reg:squarederror',

            n_jobs = -1

        )

        

        # Train model

        selection_model.fit(select_X_train, y_train)



        # Eval model - select same features as in select_X_train as well in select_X_test

        select_X_test = selection.transform(X_test)

        y_pred = selection_model.predict(select_X_test)

        selection_model_r2 = model_r2(y_test, y_pred)

        print("Thresh = {:.7f}, n = {}, R2 = {:.5f}".format(thresh, select_X_train.shape[1], selection_model_r2))



        # Append the results to a r2_feat_importance for consolidation            

        new_entry = [thresh, select_X_train.shape[1], selection_model_r2]

        r2_feat_importance.loc[len(r2_feat_importance)] = new_entry

    else:

        continue

                

end = time.time()

print('Time taken to run:', (end-start)/60)



# Show final 'r2_feat_importance' table

print(r2_feat_importance)
# Plot a graph to see the performance of XGB for different number of features

plt.subplots(figsize=(15, 10))

plt.scatter(x = r2_feat_importance['no_features'], y = r2_feat_importance['threshold_r2'], s = 5)

plt.xlabel('No. of Features in XGB')

plt.ylabel('R2 - Performance of XGB')

plt.show()
row_max_r2 = r2_feat_importance[r2_feat_importance['threshold_r2'] == r2_feat_importance['threshold_r2'].max()]

print(row_max_r2)



# Number of features for min rmse

no_features_max_r2 = row_max_r2['no_features'].values[0]

print('No. of features for max. R2 score:', no_features_max_r2)



# Corresponding threshold

threshold_max_r2 = row_max_r2['thresholds'].values[0]

print('Threshold value for max. R2 score: {:.6f}'.format(threshold_max_r2))
# We use k-fold cross validation

k_folds = 10



### Retrain the XGB model using X features only on FULL TRAINING DATA



# Modify the function for calculating mean RMSE

def r2_model_feat_impt(model):

    kf_cv = KFold(k_folds, shuffle = True, random_state = 123).get_n_splits(select_train)

    r2 = cross_val_score(

            model,

            select_train,

            y_train,

            scoring = "r2",

            cv = kf_cv

    )

    return(r2)



start = time.time()

selection = SelectFromModel(xgb, threshold = threshold_max_r2, prefit = True)

select_train = selection.transform(X_train)

select_test = selection.transform(X_test)

            

# Define model again

selection_model = XGBRegressor(

    colsample_bytree=0.2,

    gamma=0.0,

    learning_rate=0.01,

    max_depth=5, 

    min_child_weight=1.5,

    n_estimators=4000,

    reg_alpha=0.9,

    reg_lambda=0.6,

    subsample=0.8,

    verbosity=0,

    random_state = 7,

    objective='reg:squarederror',

    n_jobs = -1

)



# KFolds CV:

CV_r2 = r2_model_feat_impt(selection_model)



# Print result

print('Mean R2 of XGB training using {} features is {:.6f}'.format(no_features_min_r2, CV_r2.mean()))



end = time.time()

print('Time taken to complete {:.2f} mins'.format((end-start)/60))
start = time.time()

grid_params = {

    'learning_rate':[0.001, 0.01],

    'n_estimators':[4000, 1000],

    'reg_alpha': [0.3, 0.6, 0.9],

    'reg_lambda': [0.3, 0.6, 0.9],

    'max_depth': [3, 5, 7],

    'subsample': [0.6, 0.8]

}



GS_xgb = GridSearchCV(

    estimator = XGBRegressor(

        min_child_weight=1.5,

        colsample_bytree=0.2,

        gamma=0.0,

        verbosity=0,

        random_state = 7,

        objective='reg:squarederror',

        n_jobs = -1

    ),

    param_grid = grid_params,

    scoring='neg_mean_squared_error',

    n_jobs=-1,

    iid=False,

    cv=5,

    verbose = 1

    )



# Select_train is the full training set with just 21 feature columns

GS_xgb.fit(select_train,y_train)



#GS_gradboost_1.grid_scores_,

bestparam = GS_xgb.best_params_

bestscore = GS_xgb.best_score_



print('bestscore is:', bestscore)

print('bestparam is:', bestparam)

end = time.time()

print('Time taken to run is:', end - start)
### Retrain model again with best parameters from GridSearchCV



final_xgb = XGBRegressor(

    learning_rate=bestparam['learning_rate'],

    max_depth=bestparam['max_depth'], 

    subsample=bestparam['subsample'],

    n_estimators=bestparam['n_estimators'],

    reg_alpha=bestparam['reg_alpha'],

    reg_lambda=bestparam['reg_lambda'],

    min_child_weight=1.5,

    colsample_bytree=0.2,

    gamma=0.0,

    verbosity=0,

    random_state = 7,

    objective='reg:squarederror',

    n_jobs = -1

)



final_xgb.fit(select_train, y_train)



# Churn out the predictions based on this final model

final_xgb_pred = final_xgb.predict(select_test)



# Calculate RMSE 

final_xgb_r2 = model_r2(y_test, final_xgb_pred)

print('R2 of final XGBRegressor is: {:.6f}'.format(final_xgb_r2))