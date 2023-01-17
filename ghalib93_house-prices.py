# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing necessary libraries

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import math 

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn import metrics



%matplotlib inline

df_train_data_out = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# Reading Data from CSV to DF - Jupyter

# df_train_data_out = pd.read_csv('data/train.csv')

# df_test_data = pd.read_csv('data/test.csv')
submission = pd.DataFrame(df_test_data['Id'], columns={'Id'})
# A function that prints the top columns with missing NaN count and percent

def print_nan_percentage(df_to_print):

    # Getting the sum of missing values from DF Sorted in descending order

    nan_count = df_to_print.isnull().sum().sort_values(ascending=False)

    # Dividing the NaN sum with coulmn length to get a percentage 

    nan_percentage = nan_count / len(df_to_print)

    # Returning the top 20 columns with missing NaNs

    return pd.DataFrame(data=[nan_count, nan_percentage],index=['nan_count', 'nan_percentage']).T.head(20)
print_nan_percentage(df_train_data_out)
print_nan_percentage(df_test_data)
df_train_data_out.describe()
df_test_data.describe()
# Set index in both DFs to be the Id column since the Id between the two DFs are connected 

# Train end at 1459 and test start from 1460

df_test_data.set_index('Id')

df_train_data_out.set_index('Id')
#Set the default subplots number of rows,number of columns and figure size,:

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 10),  sharex=True)



# train data 

sns.heatmap(df_train_data_out.isnull(), yticklabels=False, ax = ax[0],cbar=False, cmap='YlGnBu')

# Set the heatmap title

ax[0].set_title('Missing data in the train dataframe')



# test data

sns.heatmap(df_test_data.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='YlGnBu')

# Set the heatmap title

ax[1].set_title('Missing data in the test dataframe');

plt.tight_layout()
print('MSZoning: ', df_train_data_out['MSZoning'].unique())

print('\nMSSubClass: ', df_train_data_out['MSSubClass'].unique())

print('\nPoolQC: ', df_train_data_out['PoolQC'].unique())

print('\nUtilities: ', df_train_data_out['Utilities'].unique())
print('MSZoning: ', df_test_data['MSZoning'].unique())

print('\nMSSubClass: ', df_test_data['MSSubClass'].unique())

print('\nPoolQC: ', df_test_data['PoolQC'].unique())

print('\nUtilities: ', df_test_data['Utilities'].unique())
#Correlation with output variable

df_corr = df_train_data_out.corr()

cor_target = abs(df_corr["SalePrice"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.4]

relevant_features
# Plotting the best correlated features with SalePrice using pair plot

sns.pairplot(df_train_data_out,height=2, x_vars=['GrLivArea',

                                             'OverallQual',

                                             'YearBuilt',

                                             'YearRemodAdd',

                                             'TotalBsmtSF',

                                             '1stFlrSF',

                                             'FullBath',

                                             'TotRmsAbvGrd',

                                             'GarageCars',

                                             'GarageArea'], y_vars='SalePrice', markers=['+'])
#box plot overallqual/saleprice

data = pd.concat([df_train_data_out['SalePrice'], df_train_data_out['OverallQual']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)

fig.axis(ymin=0, ymax=850000);
#box plot YearBuilt/saleprice

data = pd.concat([df_train_data_out['SalePrice'], df_train_data_out['YearBuilt']], axis=1)

fig, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=data, linewidth=1)

plt.xticks(rotation=90);

fig.axis(ymin=0, ymax=800000);
# Dropping outliers from train DF for column GrLivArea

df_train_data = df_train_data_out.drop(df_train_data_out[(df_train_data_out['GrLivArea']>4000) & 

                                        (df_train_data_out['SalePrice']<300000)].index)



df_train_data
gr=sns.boxplot(x=df_train_data['GrLivArea'])
# Getting the SalePrice in a target column

y = df_train_data['SalePrice']
# Dropping SalePrice from train DF(

df_train_data.drop('SalePrice',axis=1 ,inplace= True)
# Merging both train and test DF and preparing for data cleaning 

data_merge = pd.merge(df_train_data ,df_test_data,how='outer',left_index=False, right_index=False)
# Checking that the merge done appropriately 

data_merge.head()
# Replace NaNs with 0 for garage year built

data_merge['GarageYrBlt'].fillna(0, inplace=True)



# For garages with no built year, assume that it was built in the same year as the house 

fixed_garage = [j if i == 0 else i for i,j in zip(data_merge['GarageYrBlt'], data_merge['YearBuilt'])]

    

# Assigne new replaced zero values to the garage year built

data_merge['GarageYrBlt'] = fixed_garage
# Getting the columns that should be filled using the most common category in the column

col_mode_fill = ['MSZoning', 'Utilities', 'Exterior1st', 'MasVnrType', 'BsmtQual', 'KitchenQual', 

                 'Functional', 'Electrical']

# Filling the category columns with the mode of the column

data_merge.update(data_merge[col_mode_fill].fillna(df_train_data[col_mode_fill].mode(), inplace=True))

data_merge[col_mode_fill].head(100)
# Fill NaN values with NA for object columns that were not filled with mode,

# and fill floats with medain

data_merge = data_merge.apply(lambda x: x.fillna('NA') if x.dtype.kind in 'O' else x.fillna(x[:1457].median()) 

                               if x.dtype.kind in 'f' else x)

data_merge.head()
# Display train info to check that data does not have any more NaN values

print_nan_percentage(data_merge)
# Creating a heatmap for Merged Data 

sns.heatmap(data_merge.isnull(), yticklabels=False,cbar=False, cmap='YlGnBu')

# Set the heatmap title

plt.title('Missing data in merged dataframe')

data_merge.set_index("Id", inplace=True)
# Import skew to calculate the skew of the functions and BoxCox for transformation

from scipy.stats import skew

from scipy.special import boxcox1p



# Get numerical columns from DF

numeric_feats = data_merge.dtypes[data_merge.dtypes != "object"].index



# Check the skew of all numerical features

skewed_cols = data_merge[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_cols})

skewness.head(10)
# Check how many columns are skewed based on previous scores 

skewness = skewness[abs(skewness) > 0.75]

print("There are ", skewness.shape[0], " skewed numerical features to Box Cox transform")
# Get the index of skewed features

skewed_features = skewness.index



lam = 0.15

for feat in skewed_features:

    data_merge[feat] = boxcox1p(data_merge[feat], lam)
# Convert categorical variable into dummy/indicator variables.

dt_me_dumy = pd.get_dummies(data_merge, drop_first=True)
dt_me_dumy.info()
train_cleaned = dt_me_dumy[:1458]
train_cleaned.tail()
test_cleand = dt_me_dumy[1458:]
test_cleand.head()
# initialize the Scaler

ss = StandardScaler()



# Fit train data using the scaler (scale the data)

train_cleaned_s = ss.fit_transform(train_cleaned)



test_cleand_s = ss.transform(test_cleand)
# Display standardized data (train)

train_cleaned_s
# Display standardized data (test)

test_cleand_s
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
sns.distplot(y, norm_hist=True, bins=146)
ylog = y.copy()

ylog = np.log(ylog)

sns.distplot(ylog, norm_hist=True, bins=146)
# A function that takes a model name, its scores, the train score and prints them

def print_model_mean_and_train(model, scores, train_scores):

    print(model, " mean score is:", scores.mean())

    print('Train score for ', model,' is: ', train_scores)
# a function that takes the predictions of the model, the model name and save ot to a CSV

# The function will return the first 10 predictions

def save_csv(predictions, model_name):

    # Reverse log operation on predictions using exp

    submission['SalePrice'] = np.exp(predictions)

    # Save CSV to path

    submission.to_csv('submission_'+model_name+'.csv', index=False)

    # Jupyter save to csv

    #submission.to_csv('data/submission_'+model_name+'.csv', index=False)

    return submission.head(10)
# A function that takes the model, X & y train, cv number, and performs cross validation

# The function will return the model and its scores

def performe_CV(used_model, X, y_data, cv):

    used_model.fit(X, y_data)

    scores = cross_val_score(used_model, X, y_data, cv=cv)

    return used_model, scores
from sklearn.feature_selection import SelectFromModel
# Build LassoCV regression model 



# Fit on standarized data and performe 8-fold CV

lassoCV, lassoCV_scores = performe_CV(LassoCV(), train_cleaned_s, ylog, 8)



# Print Model Mean Score

print_model_mean_and_train('LassoCV', lassoCV_scores, lassoCV.score(train_cleaned_s, ylog))



# Get best alpha

bestAlphaLasso = lassoCV.alpha_

print("Best Alpha for LassoCV: ", bestAlphaLasso)
# Build Lasoo regression model using best alpha

lasso = Lasso(alpha=bestAlphaLasso, copy_X=True,

                             fit_intercept=True, max_iter=1000, normalize=False,

                             positive=False, precompute=False,

                             random_state=None, selection='cyclic', tol=0.0001,

                             warm_start=False)

   

# Fit on standarized data and performe 8-fold CV

lasso, lasso_scores = performe_CV(lasso, train_cleaned_s, ylog, 8)

# Print Model Mean Score

print_model_mean_and_train('Lasso', lasso_scores, lasso.score(train_cleaned_s, ylog))
# Give the model data that it have not seen yet and get the predictions

pred_lasso = lasso.predict(test_cleand_s)

pred_lasso
save_csv(pred_lasso, 'Lasso')
sfm = SelectFromModel(lasso, threshold=0.1)

sfm.fit(train_cleaned_s, ylog)

features = sfm.transform(train_cleaned_s)

feature_idx = sfm.get_support()

feature_name = train_cleaned.columns

#feature_name = feature_name.drop('Id')

feature_name
# Get a copy of the best features and standardize it 

best_train = train_cleaned



# Dropping multicollinearity columns

best_train.drop('1stFlrSF', axis=1, inplace=True)

best_train.drop('TotRmsAbvGrd', axis=1, inplace=True)                                             

best_train.drop('GarageArea', axis=1, inplace=True)                                             

best_train
# Get a copy of the best features and standardize it 

best_test = test_cleand



# Dropping multicollinearity columns

best_test.drop('1stFlrSF', axis=1, inplace=True)

best_test.drop('TotRmsAbvGrd', axis=1, inplace=True)                                             

best_test.drop('GarageArea', axis=1, inplace=True)   
# Fit train data using the scaler (scale the data)

best_train_s = ss.fit_transform(best_train)

best_test_s = ss.transform(best_test)
from sklearn.model_selection import cross_val_score
# Create model instance

lm = LinearRegression()

     

# Perform 10-fold cross validation and fit on model

lm, lm_scores = performe_CV(lm,  best_train, ylog, 10)

print_model_mean_and_train('Linear Regression', lm_scores, lm.score(best_train, ylog))

# Give the model data that it have not seen yet and get the predictions

pred_lm = lm.predict(best_test)

pred_lm
save_csv(pred_lm, 'LinearReg')
# np.logspace gives us points between specified orders of magnitude on a logarithmic scale. It is base 10.

r_alphas = np.logspace(0, 5, 500)
# Build RidgeCV regression model 



# Perform 8-fold cross validation and fit on model

ridgeCV, ridgeCV_scores = performe_CV(RidgeCV(alphas=r_alphas), best_train, ylog, 8)

print_model_mean_and_train('RidgeCV', ridgeCV_scores, ridgeCV.score(best_train, ylog))        



# Get best alpha 

bestAlpha = ridgeCV.alpha_

print("Best Alpha for RidgeCV: ", bestAlpha)
# Build Ridge regression model using best alpha

ridge = Ridge(alpha=bestAlpha, copy_X=True, fit_intercept=True, max_iter=None,

      normalize=False, random_state=1, solver='auto', tol=0.001)



# Perform 8-fold cross validation and fit on model

ridge_model, ridge_scores = performe_CV(ridge, best_train, ylog, 8)

print_model_mean_and_train('Ridge', ridge_scores, ridge_model.score(best_train, ylog))
# Give the model data that it have not seen yet and get the predictions

pred_ridge = ridge_model.predict(best_test)

pred_ridge
save_csv(pred_ridge, 'Ridge')
# Creating multiple distplots to compare the predictions with the target data

plt.figure(figsize=(20,10))

sns.distplot(pred_ridge, label='Ridge', hist=False)

sns.distplot(pred_lasso, label='Lasso', hist=False)

sns.distplot(pred_lm, label='LinearReg', hist=False)

sns.distplot(ylog, label='Log Y', hist=False)

plt.legend(fontsize='22')
# Creating l1_ratio points to try and get optimal ratio 

l1_ratios = np.linspace(0.01, 1.0, 5)



# Build ElasticNetCV regression model

elasticCV = ElasticNetCV(l1_ratio=[.1, .5, .7,

    .9, .95, .99, 1], n_alphas=600)



# Perform 8-fold cross validation and fit on model

elasticCV, elasticCV_scores = performe_CV(elasticCV, best_train_s, ylog, 8)

print_model_mean_and_train('ElasticNetCV', elasticCV_scores, elasticCV.score(best_train_s, ylog))





# Getting best alpha and optimal ratio

bestAlphaElastic = elasticCV.alpha_

optimall1 = elasticCV.l1_ratio_



print("Best Alpha for ElasticNetCV: ", bestAlphaElastic)

print("Optimal l1_ratio for ElasticNetCV: ", optimall1)
# Build ElasticNet regression model

elasticNet = ElasticNet(alpha=bestAlphaElastic, l1_ratio=optimall1, random_state=1)



# Perform 8-fold cross validation and fit on model

elastic, elastic_scores = performe_CV(elasticNet, best_train_s, ylog, 8)

print_model_mean_and_train('ElasticNet', elastic_scores, elastic.score(best_train_s, ylog))
# Give the model data that it have not seen yet and get the predictions

pred_elastic = elastic.predict(best_test_s)

pred_elastic
save_csv(pred_elastic, 'elastic')
from sklearn.model_selection import GridSearchCV
# Build a GS regression model 



# Creating a list of alphas for Ridge

alphas = np.logspace(-4, -0.5, 30)



# Setting the parameters for Ridge Grid Search

tuned_parameters = {'alpha': alphas,

                    'fit_intercept': [True,False], 

                    'normalize' :[False, True]}



# Performing GS CV 

gs = GridSearchCV(ridge, param_grid=tuned_parameters, cv=5)



# Fitting the model 

gs.fit(best_train_s, ylog)



# Printing the score 

print("Grid Search Train Score is: ",gs.score(best_train_s, ylog))
# Give the model data that it have not seen yet and get the predictions

gs_pred = gs.predict(best_test_s)

gs_pred
# Getting the best estimator from the model 

best = gs.best_estimator_

best
# Getting the best Parameters from the model 

gs.best_params_
save_csv(gs_pred, 'grid')
from sklearn.ensemble import RandomForestRegressor
# Building RF model

rf = RandomForestRegressor()



# Perform 7-fold cross validation and fit on model

rf, rf_score = performe_CV(rf, best_train_s, ylog, 7)

print_model_mean_and_train('RF', rf_score, rf.score(best_train_s, ylog))
# Give the model data that it have not seen yet and get the predictions

pred_rf = rf.predict(best_test_s)

pred_rf
save_csv(pred_rf, 'RF')