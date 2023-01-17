# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Reading the data

train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
# Lets take a look at the first few rows of the dataset

train_df.head()
# Lets look at the shape

train_df.shape
# Setting the grid style

sns.set_style('darkgrid')

sns.set_color_codes(palette='dark')



# Setting plot area

f, ax = plt.subplots(figsize=(9, 9))



# plotting the distribution plot

sns.distplot(train_df['SalePrice'], color="m", axlabel='SalePrice')

ax.set(title="Histogram for SalePrice")

plt.show()
# Calc correlation matrix

corr_mat = train_df.corr()



# Set plot size

plt.subplots(figsize=(12,10))



# Plot heatmap

sns.heatmap(corr_mat, 

            square=True, 

            robust=True, 

            cmap='OrRd', # use orange/red colour map

            cbar_kws={'fraction' : 0.01}, # shrink colour bar

            linewidth=1) # space between cells
# number of variables we want on the heatmap

k = 10 



# Filter in the Top k variables with highest correlation with SalePrice

cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)



cmap_ch = sns.cubehelix_palette(as_cmap=True, light=.95)

# Creating the heatmap

hm = sns.heatmap(cm,

                 cmap = cmap_ch,

                 cbar = True, 

                 annot = True, # Since we want to know the the correlation coeff as well.

                 square = True, 

                 robust = True,

                 cbar_kws={'fraction' : 0.01}, # shrink colour bar

                 annot_kws={'size': 8}, # setting label size

                 yticklabels=cols.values, # set y labels

                 xticklabels=cols.values,

                 linewidth=1) # Set xlabels

plt.show()
# Creating a boxplot

chart = sns.catplot(data = train_df ,

                    x = 'OverallQual',

                    y='SalePrice',

                    kind='box',

                    height=8,

                    palette='Set2')



# Setting X axis labels

chart.set_xticklabels(fontweight='light',fontsize='large')
# Creating a boxplot

chart = sns.catplot(data = train_df ,x = 'YearBuilt',y='SalePrice',

                    kind='box', # we want a boxplot

                    height=5,

                    aspect=4,

                    palette='Set2')



# Setting X axis labels

chart.set_xticklabels(fontweight='light',

                      fontsize='large',

                      rotation=90,

                      horizontalalignment='center')
plt.figure(figsize=(10,8))



# Creating a scatterplot

sns.scatterplot(data = train_df ,

                x = 'TotalBsmtSF',

                y ='SalePrice',

                alpha = 0.65,

                color = 'g') 
plt.figure(figsize=(10,8))



# Creating a scatterplot

sns.scatterplot(data = train_df ,

                x = 'GrLivArea',

                y ='SalePrice',

                alpha = 0.65,

                color = 'b') 
# Creating a boxplot

chart = sns.catplot(data = train_df ,

                    x = 'GarageCars',

                    y='SalePrice',

                    kind='box',

                    height=6,

                    palette='Set2')



# Setting X axis labels

chart.set_xticklabels(fontweight='light',fontsize='large')
# Storing the IDs in a separate DF

train_df_IDs = train_df['Id']

test_df_IDs = test_df['Id']



# Dropping the columns

train_df.drop(['Id'], axis=1, inplace=True)

test_df.drop(['Id'], axis=1, inplace=True)



# Checking the shape of both DFs

print(train_df.shape) 

print(test_df.shape)
# Log transforming SalePrice

train_df["SalePrice_log"] = np.log(train_df["SalePrice"])



# Plotting to vizualize the transformed variable

sns.distplot(train_df['SalePrice_log'], color="m", axlabel='SalePrice_log')
# Dropping SalePrice_log as we will clean it in the next steps with pipeline

train_df = train_df.drop('SalePrice_log',axis=1)
# Dropping the outliers

train_df.drop(train_df[(train_df['GrLivArea']>4500) & (train_df['SalePrice']<300000)].index, inplace=True)



# Vizualizing the new scatterplot

plt.figure(figsize=(10,8))



# Creating a scatterplot

sns.scatterplot(data = train_df ,

                x = 'GrLivArea',

                y ='SalePrice',

                alpha = 0.65,

                color = 'b') 
# Separating Predictor and Labels

housing = train_df.drop("SalePrice",axis=1)

housing_labels = train_df['SalePrice']
# calc total missing values

total_series = housing.isnull().sum().sort_values(ascending=False)



# calc percentages

perc_series = (housing.isnull().sum()/housing.isnull().count()).sort_values(ascending = False)



# concatenating total values and percentages

missing_data = pd.concat([total_series, perc_series*100], axis=1, keys=['Total #', 'Percent'])



# Looking at top 20 entries

missing_data.head(20)
# converting numeric columns to categorical

cols_int_to_str = ['MSSubClass','YrSold','MoSold','GarageYrBlt']



for col in cols_int_to_str:

    housing[col] = housing[col].astype(str)

    test_df[col] = test_df[col].astype(str)
# Creating a list of numerics we want for the mask

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



# Creating a dataframe for numeric features

housing_num = housing.select_dtypes(include=numerics)

print(housing_num.shape)
# List of Categorical features that are to be filled with None

cat_none = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond',\

            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

# Creating a dataframe of features for "none"

housing_cat_none = housing[cat_none]



# All the other categorical features

housing_cat_freq = housing[(housing.columns.difference(cat_none)) & (housing.columns.difference(housing_num.columns))]

# Importing the modules

from sklearn.base import BaseEstimator, TransformerMixin



# getting index of relevant columns instead of hardcoding

BsmtFinSF1_ix, BsmtFinSF2_ix, flr_1_ix, flr_2_ix,\

FullBath_ix, HalfBath_ix, BsmtFullBath_ix, BsmtHalfBath_ix,\

OpenPorchSF_ix, SsnPorch_ix, EnclosedPorch_ix, ScreenPorch_ix, WoodDeckSF_ix = [

    list(housing_num.columns).index(col)

    for col in ("BsmtFinSF1", "BsmtFinSF2","1stFlrSF","2ndFlrSF",\

                "FullBath","HalfBath","BsmtFullBath","BsmtHalfBath",\

                "OpenPorchSF","3SsnPorch","EnclosedPorch","ScreenPorch","WoodDeckSF")]



# Creating CombinedAttributesAdder class for creating the features

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, Total_sqr_footage = True, Total_Bathrooms=True,Total_porch_sf=True): 

        self.Total_sqr_footage = Total_sqr_footage

        self.Total_Bathrooms = Total_Bathrooms

        self.Total_porch_sf = Total_porch_sf

        

    def fit(self, X, y=None):

        return self 

    

    def transform(self, X, y=None):

        if self.Total_sqr_footage: # Calculate total footage

            Total_sqr_footage = X[:, BsmtFinSF1_ix] + X[:, BsmtFinSF2_ix] + X[:,flr_1_ix] + X[:,flr_2_ix]

       

        if self.Total_Bathrooms: # Calculate total bathrooms

            Total_Bathrooms = X[:, FullBath_ix] + X[:, HalfBath_ix] + X[:,BsmtFullBath_ix] + X[:,BsmtHalfBath_ix]

            

        if self.Total_porch_sf: # Calculate total porch area

            Total_porch_sf = X[:, OpenPorchSF_ix] + X[:, SsnPorch_ix] + X[:,EnclosedPorch_ix] + X[:,ScreenPorch_ix] + X[:,WoodDeckSF_ix]

            

        return np.c_[X, Total_sqr_footage,Total_Bathrooms,Total_porch_sf]

    
# Importing necessary libraries

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



# Creating numerical pipeline

num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="mean")), # To impute na values with mean 

        ('attribs_adder', CombinedAttributesAdder()), # Create new features

        ('std_scaler', StandardScaler()), # Scale the numeric features

    ])
# Importing necessary libraries

from sklearn.preprocessing import OneHotEncoder



# Creating pipeline for categorical variables with missing value should be "None"

cat_pipeline_none = Pipeline([

        ('imputer', SimpleImputer(strategy='constant',fill_value='None')),

        ('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'))

    ])



# Creating pipeline for categorical variables where we plug missing value with most frequent value

cat_pipeline_freq = Pipeline([

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'))

    ])
# Importing ColumnTransformer

from sklearn.compose import ColumnTransformer



# Creating full pipeline to process numeric and categorical features

full_pipeline = ColumnTransformer(transformers=[

        ("num", num_pipeline, housing_num.columns),

        ("cat_none", cat_pipeline_none, housing_cat_none.columns),

        ("cat_freq", cat_pipeline_freq, housing_cat_freq.columns),

    ])



# Instatiating the full pipelines object

transf = full_pipeline.fit(housing)



# Creating prepared data by passing the training set without labels

housing_prepared = transf.transform(housing)
# Checking the shape of the newly created data

housing_prepared.shape
# Cleaning the test data 

test_prepared = transf.transform(test_df)
# Checking Test data's shape

test_prepared.shape
# Defining CV Score function we will use to calculate the scores

def cv_score(score):

    rmse = np.sqrt(-score) # -score because we are using "neg_mean_squared_error" as our metric

    return (rmse)
%%time



# Import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error as MSE



# Instantiate RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)



# Creating Parameter grid for GridSearch CV

params_rf = {

    'n_estimators': [500,1000], # No of trees

    'max_depth': [10,15], # maximum depth to explore

    'min_samples_split':[5], # minimum samples required for split

    'min_samples_leaf':[5], # minimum samples required at leaf

    'max_features': [ 'auto'] # number of features for the best split

}



# Instantiate grid_rf

grid_rf = GridSearchCV(estimator = rf, # regressor we want to use

                       param_grid = params_rf, # Hyperparameter space

                       scoring ='neg_mean_squared_error', # MSE will be performance metric

                       cv = 3, # #of folds

                       verbose = 1,

                       n_jobs = -1) # use all cores



# fit the model

grid_rf.fit(housing_prepared,housing_labels)
# Lets look at the Cross Validation score for RandomForestRegressor

print('CV Score for best RandomForestRegressor model: {:.2f}'.format(cv_score(grid_rf.best_score_)))
# Store the best model 

best_model_RF = grid_rf.best_estimator_
%%time



# Importing the necessary package

from sklearn.ensemble import GradientBoostingRegressor



# Instantiate the Gradient Boosting Regressor

gbr = GradientBoostingRegressor(subsample = 0.9, # this is essentially stochastic gradient boosting

                                max_features = 0.75,

                                random_state = 42,

                                warm_start = True,

                                learning_rate= 0.01) # low learning rate



# Creating Parameter grid for GridSearch CV

params_gbr = {

    'n_estimators': [8000], # Given that the learning rate is very low, we are increasing the num of estimators

    'max_depth': [2,3], 

    'min_samples_split':[5],

    'min_samples_leaf':[5],

    'max_features': ['sqrt']

}



# Instantiate grid search using GradientBoostingRegressor

grid_gbr = GridSearchCV(estimator = gbr, # regressor we want to use

                       param_grid = params_gbr, # Hyperparameter space

                       scoring ='neg_mean_squared_error',

                       cv = 3, # No of folds

                       verbose = 1,

                       n_jobs = -1) # use all cores



# fit the model

grid_gbr.fit(housing_prepared,housing_labels)
# Lets look at the Cross Validation score for GradientBoostingRegressor

print('CV Score for best GradientBoostingRegressor model: {:.2f}'.format(cv_score(grid_gbr.best_score_)))
# Store the best model 

best_model_GBR = grid_gbr.best_estimator_
%%time



from sklearn.linear_model import Ridge



# Instantiate the Ridge Regressor

ridge = Ridge(random_state=42)



# Creating Parameter grid for GridSearch CV

params_ridge = {

    'alpha': [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100], #L2 parameter space

    'solver': ['auto','saga','sag','cholesky']

}



# Instantiate grid search using GradientBoostingRegressor

grid_ridge = GridSearchCV(estimator = ridge, # regressor we want to use

                       param_grid = params_ridge, # Hyperparameter space

                       scoring ='neg_mean_squared_error',

                       cv = 3, # No of folds

                       verbose = 1,

                       n_jobs = -1) # use all cores



# fit the model

grid_ridge.fit(housing_prepared,housing_labels)

# Lets look at the Cross Validation score for RidgeRegressor

print('CV Score for best RidgeRegressor model: {:.2f}'.format(cv_score(grid_ridge.best_score_)))
# Store the best model 

best_model_ridge = grid_ridge.best_estimator_
%%time

from sklearn.linear_model import ElasticNet



# Instantiate the Ridge Regressor

elastic = ElasticNet(random_state=42)



# Creating Parameter grid for GridSearch CV

params_elastic = {

    'alpha': [ 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15], #L2 regularization parameter space

    'l1_ratio': [0.01,0.1,0.3,0.5,0.8] #L1 regularization parameter space

}



# Instantiate grid search using GradientBoostingRegressor

grid_elastic = GridSearchCV(estimator = elastic, # regressor we want to use

                       param_grid = params_elastic, # Hyperparameter space

                       scoring ='neg_mean_squared_error',

                       cv = 3, # No of folds

                       verbose = 1,

                       n_jobs = -1) # use all cores



# fit the model

grid_elastic.fit(housing_prepared,housing_labels)

# Lets look at the Cross Validation score for ElasticNet

print('CV Score for best ElasticNet model: {:.2f}'.format(cv_score(grid_elastic.best_score_)))
# Store the best model 

best_model_elastic = grid_elastic.best_estimator_
%%time

from sklearn.svm import SVR



# Instantiate the SVM Regressor

svr = SVR()



# Creating Parameter grid for GridSearch CV

params_svr = {

    'kernel': ['poly'], # we want a polynomial kernel

    'degree': [5,8], # degrees to test

    'gamma':[0.01,0.05], 

    'epsilon': [1.5,3], 

    'coef0':[3,5], # since we are selecting a polynomial kernel

    'C': [10,30], # Penalty parameter

    'tol':[1e-7,1e-5] # Tolerance for stopping

}



# Instantiate grid search using GradientBoostingRegressor

grid_svr = GridSearchCV(estimator = svr, # regressor we want to use

                       param_grid = params_svr, # Hyperparameter space

                       scoring ='neg_mean_squared_error',

                       cv = 3, # No. of folds

                       verbose = 1,

                       n_jobs = -1) # use all cores



# fit the model

grid_svr.fit(housing_prepared,housing_labels)

# Lets look at the Cross Validation score for SVM

print('CV Score for best SVM model: {:.2f}'.format(cv_score(grid_svr.best_score_)))
# Store the best model 

best_model_svr = grid_svr.best_estimator_
%%time

import xgboost as xgb



# Instantiate the SVM Regressor

xgbr = xgb.XGBRegressor(learning_rate=0.01,objective='reg:linear',booster='gbtree')



# Creating Parameter grid for GridSearch CV

params_xgb = {

    'n_estimators': [8000,10000], #4000,12000

    'max_depth': [2],

    'gamma':[0.1,0.2], # Minimum loss reduction to create new tree split ,0.5,0.9

    'subsample':[0.7], 

    'reg_lambda':[0.1], 

    'reg_alpha':[0.1,0.8] 

}



# Instantiate grid search using GradientBoostingRegressor

grid_xgb = GridSearchCV(estimator = xgbr, # regressor we want to use

                       param_grid = params_xgb, # Hyperparameter space

                       scoring ='neg_mean_squared_error',

                       cv = 3, # No. of folds

                       verbose = 1,

                       n_jobs = -1) # use all cores



# fit the model

grid_xgb.fit(housing_prepared,housing_labels)

# Lets look at the Cross Validation score for XGBRegressor

print('CV Score for best XGBRegressor model: {:.2f}'.format(cv_score(grid_xgb.best_score_)))
# Store the best model 

best_model_xgb = grid_xgb.best_estimator_
# importing Voting Regressor

from sklearn.ensemble import VotingRegressor



# Instantiate the Regressor

voting_reg = VotingRegressor(

    estimators=[('rf', best_model_RF), ('gbr', best_model_GBR), ('elastic', best_model_elastic),

               ('ridge',best_model_ridge),('svr',best_model_svr),('xgb',best_model_xgb)])
# Importing cross validation module

from sklearn.model_selection import cross_val_score



# Calculate cross validation score for the Voting regresssor

scores = cross_val_score(voting_reg, housing_prepared, housing_labels,

                         scoring="neg_mean_squared_error", cv=10)
# Given that we have negative MSE score, lets first get the root squares to get RMSE's and then calculate the mean

voting_reg_score = np.sqrt(-scores)



# Calc mean for RMSE

print(voting_reg_score.mean())
# Fitting the voting regressor on the entire training dataset

voting_reg.fit(housing_prepared, housing_labels)



# Predict on test set

pred = voting_reg.predict(test_prepared)
# converting to dataframe

preds_df = pd.DataFrame({'Id':test_df_IDs,'SalePrice':pred},index=None)
# looking at the first 5 rows

preds_df.head()
# Submitting the predictions

preds_df.to_csv('submissions.csv',index=False)