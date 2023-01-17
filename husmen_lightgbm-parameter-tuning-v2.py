# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#plotting

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style("whitegrid")



#ML

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split # Model evaluation

from lightgbm import LGBMRegressor, plot_importance # LGBM



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import datetime

import time

import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load data and display info dataset

df_train = pd.read_csv("../input/train.csv", index_col = 0) 

df_test = pd.read_csv("../input/test.csv", index_col = 0) 
df_train.info()
#display Missing values

label_nas = []

for col in df_train.columns.tolist():

    if np.sum(df_train[col].isnull() ) != 0:

        label_nas.append(col)

    else:

        label_nas.append("")



plt.figure(figsize=(12,7))

plt.suptitle('Missing Values in the Training Set')

sns.heatmap(df_train.isnull(), yticklabels=False, xticklabels=label_nas ,cbar = False, cmap='viridis')

plt.show()
#remove columns with too many missed values

null_values_per_col = np.sum(df_train.drop(["SalePrice"], axis=1).isnull(), axis=0)

max_na = int(df_train.shape[0]/3.0) #allowing up to 1/3 of the data to be missing

cols_to_remove = []



for col in df_train.drop(["SalePrice"],axis=1).columns.tolist():

    if null_values_per_col[col] > max_na: 

        cols_to_remove.append(col)

        df_train.drop(col, axis=1, inplace=True)

        

print("New shape of the training set is: (%d,%d)" %df_train.shape)        

print("The removed columns are: " + str(cols_to_remove))



#do the same for test dataset

df_test.drop(cols_to_remove, axis=1, inplace=True)
# MSSubClass is actually categorical even if expressed as numerical

df_train = df_train.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"}})

#remap some columns to have numerical data

rep_map_1 = {"NA": 0, "Po": 1,

             "Fa": 2, "TA": 3, 

             "Gd": 4, "Ex": 5,}



rep_map_2 = {"NA": 0, "No": 1,

             "Mn": 2, "Av": 3, 

             "Gd": 4,}



rep_map_3 = {"NA": 0, "Unf": 1,

             "LwQ": 2, "Rec": 3, 

             "BLQ": 4, "ALQ": 5,

             "GLQ": 6,}



rep_map_4 = {"NA": 0, "MnWw": 1,

             "GdWo": 2, "MnPrv": 3, 

             "GdPrv": 4,}



scale_list_1 = ["ExterCond", "BsmtCond", "HeatingQC", "KitchenQual", "GarageCond"]

scale_list_2 = ["BsmtFinType1"]

scale_list_3 = ["BsmtFinType2"]

scale_list_4 = ["Fence"]
for s in scale_list_1:

    df_train = df_train.replace({s:rep_map_1,})

    df_test = df_test.replace({s:rep_map_1,})



df_train = df_train.replace({scale_list_2[0]:rep_map_2,

                             scale_list_3[0]:rep_map_3,

                             scale_list_4[0]:rep_map_4,})



df_test = df_test.replace({scale_list_2[0]:rep_map_2,

                             scale_list_3[0]:rep_map_3,

                             scale_list_4[0]:rep_map_4,})
#Correlation matrix

corr_mat = df_train.corr().abs()

# Find most important features relative to target

corr_mat.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr_mat.SalePrice)
#Correlation matrix between features

corr_mat = df_train.drop(columns=["SalePrice"]).corr().abs()
#Plot the heatmap with seaborn

plt.figure(figsize=(25,25)) 

sns.heatmap(corr_mat, 

        xticklabels=corr_mat.columns,

        yticklabels=corr_mat.columns)
#List the highly correlated columns

corr_tmp = corr_mat.unstack()

corr_tmp = corr_tmp.sort_values(kind="quicksort")



print(corr_tmp[-len(corr_mat)-20:-len(corr_mat)])
#Remove highly correlated columns if needed, I will skip that

rm_corr = False

if rm_corr:

    columns_corr = ["GarageYrBlt", "GarageCars", "GrLivArea"]

    df_train = df_train.drop(columns=columns_corr)

    df_test = df_test.drop(columns=columns_corr)
#Check the skewness of SalePrice

skewness = df_train['SalePrice'].skew()

df_train['SalePrice'].plot.hist(edgecolor='white', bins=30, 

                                label='SalePrice skew =' + str(round(skewness,2)))

plt.suptitle("SalePrice distribution")

plt.legend()

plt.show()
#Apply log1p to reduce skewness

skewness = np.log1p(df_train['SalePrice']).skew()

np.log(df_train['SalePrice']).plot.hist(edgecolor='white', bins=30, 

                                label='SalePrice skew =' + str(round(skewness,2)))

plt.suptitle("SalePrice distribution")

plt.legend()

plt.show()
# Log transform the SalePrice column

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
#Get a list of all numerical and categorical features

categorical_list = []

numerical_list = []



for col in df_train.drop(columns=["SalePrice"]).columns.tolist():

    if df_train[col].dtype == 'object':

        categorical_list.append(col)

    else:

        numerical_list.append(col)
# Log transform of the skewed numerical features to lessen impact of outliers

skewness = df_train[numerical_list].skew()

skewness = skewness[abs(skewness) > 0.7]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features = skewness.index

df_train[skewed_features] = np.log1p(df_train[skewed_features])

df_test[skewed_features] = np.log1p(df_test[skewed_features])
#Convert categorical variables into dummy/indicator variables

df_train = pd.get_dummies(df_train, columns=categorical_list)

df_test = pd.get_dummies(df_test, columns=categorical_list)

df_train.info()
#If you chose to leave out some columns, remove them now

for col in df_train.columns.tolist():

    if df_train[col].dtype == 'object':

        del df_train[col]

        del df_test[col]

df_train.info()
train_stats = df_train.describe()

train_stats
#prepare for splitting and normalization

df_train_tmp = df_train.drop(columns=["SalePrice"]) #Normalize across the whole training dataset

norm = False

def norm(x):

    #return (x - train_stats['mean']) / train_stats['std']

    return (x - df_train_tmp.mean()) / (df_train_tmp.max() - df_train_tmp.min())
#Normalise, split Z and Y, split data into 80% training and 20% validation set

y_full = df_train.pop('SalePrice')

if norm:

    X_full = norm(df_train)

    X_test = norm(df_test)

else:

    X_full = df_train

    X_test = df_test



X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=123)
tune = True

#Cross validation

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

#Function to quickly recreate the model when needed

model = LGBMRegressor(objective='regression',

                          num_iterations = 2000,

                          learning_rate = 0.01,

                          bagging_freq = 5,

                          bagging_fraction = 0.8, #subsample

                          feature_fraction = 0.4, #colsample_bytree

                          device_type='cpu',

                          random_state=123)
# 1: optimise n_estimators and max_depth

# the range to explore depends on the problem, and your observations while trying 

# I already know the best parameters, I'll use that to reduce running time

# for the meaning of each parameter, check the official documentation 

# https://lightgbm.readthedocs.io/en/latest/Parameters.html

# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html



num_leaves = [11, 17] #11

min_data_in_leaf = [2] #2

max_depth = [5, 7] #5

max_bin = [47, 63] #47



# create dictionary with the parameters

param_grid = dict(num_leaves=num_leaves, max_bin=max_bin, max_depth=max_depth,

                  min_data_in_leaf=min_data_in_leaf)



print("number of combinations: {}".format(len(max_depth)*len(num_leaves)*len(max_bin)*len(min_data_in_leaf)))
#fine tune the model

if tune:

    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold, verbose=3)

    grid_result = grid_search.fit(X_full, y_full)
# summarize results

if tune:

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']

    stds = grid_result.cv_results_['std_test_score']

    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):

        print("%f (%f) with: %r" % (mean, stdev, param))
# final model with tuned parameters

model = LGBMRegressor(objective='regression',

                       num_leaves=17, #dflt 31

                       learning_rate=0.01, #dflt 0.1

                       num_iterations=2000, #n_estimators dflt 100

                       max_bin = 63, #dflt 63

                       max_depth = 5,

                       bagging_freq = 5,

                       bagging_fraction = 0.8, #subsample 0.8

                       feature_fraction = 0.4, #colsample_bytree 0.2319 

                       min_data_in_leaf = 2, #dflt = 20 

                       min_sum_hessian_in_leaf = 1e-3, #min_child_weight, dflt 1e-3

                       device_type='cpu', #gpu was slower than cpu for some reason!

                       random_state=123)
# final Cross-Validation

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

results = cross_val_score(model, X_full, y_full, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold, verbose=3)
# CV results

results_rmsle = np.sqrt(-results)

print("-MSE: {} ({})".format(results.mean(),results.std()))

print("RMS(L)E: {} ({})".format(results_rmsle.mean(),results_rmsle.std()))
#Final training

tick=time.time()



fitted_model = model.fit(X_train, y_train,

                         eval_set=[(X_val, y_val)],

                         eval_metric='rmse', # Equivalent to RMSLE since we have log1p(SalePrice)

                         early_stopping_rounds=200,

                         verbose=True)



print("Duration: {}s".format(time.time()-tick))
# Plot feature importance

figsize=(40,40)

fig, ax = plt.subplots(1,1,figsize=figsize)

plot_importance(model, ax=ax,height = 1)
# predict sales in validation dataset

val_predictions = model.predict(X_val).flatten()



# scatter plot of True vs Predicted values

plt.scatter(y_val, val_predictions)

plt.xlabel('True Values $')

plt.ylabel('Predictions $')

plt.axis('equal')

plt.axis('square')

plt.xlim([10,plt.xlim()[1]])

plt.ylim([10,plt.ylim()[1]])

_ = plt.plot([-100, 100], [-100, 100])
# Histogram of error values

error = val_predictions - y_val

plt.hist(error, bins = 25)

plt.xlabel("Prediction Error $")

_ = plt.ylabel("Count")
# predict sales for test dataset

predictions = model.predict(X_test[X_train.columns]) # keep columns in same order

# reverse log1p, and round results

predictions = np.round(np.exp(predictions)-1)

print(predictions)
# prepare dataframe to submit

submission = pd.DataFrame({'Id': df_test.index, 'SalePrice': predictions})

submission.head()
# submit

submission.to_csv('lgbm2_v2_{}.csv'.format(round(results_rmsle.mean(),5)), index=False)
!ls