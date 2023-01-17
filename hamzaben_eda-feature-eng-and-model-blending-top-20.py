# importing Python modules

import os

import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))

print("Python environment: {}".format(sys.executable))



import pandas as pd 

from pandas import ExcelWriter

from pandas import ExcelFile

#from openpyxl import load_workbook

print("pandas version: {}". format(pd.__version__))



import plotly_express as px

import matplotlib #collection of functions for scientific and publication-ready visualization

import matplotlib.pyplot as plt # for plotting

%matplotlib inline

print("matplotlib version: {}". format(matplotlib.__version__))

import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

print("seaborn version: {}". format(sns.__version__))



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics

from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

print("SciPy version: {}". format(sp.__version__)) 



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

from IPython.display import display

pd.options.display.max_columns = None

print("IPython version: {}". format(IPython.__version__)) 



import datetime

from datetime import datetime

from dateutil.parser import parse

from time import time



# to make this notebook's output identical at every run

np.random.seed(42)



print("Imported required Python packages")
# scikit-learn modules

import sklearn

print("scikit-learn version: {}". format(sklearn.__version__))

# sklearn modules for preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# from imblearn.over_sampling import SMOTE  # SMOTE

# sklearn modules for ML model selection

from sklearn.model_selection import train_test_split  # import 'train_test_split'

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# Libraries for data modelling

from sklearn import svm, tree, linear_model, neighbors

from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor # import RandomForestRegressor

from sklearn.ensemble  import AdaBoostClassifier

from sklearn.ensemble  import GradientBoostingRegressor

from sklearn.linear_model import Lasso



# Common sklearn Model Helpers

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.preprocessing import MinMaxScaler, RobustScaler

# from sklearn.datasets import make_classification



# sklearn modules for performance metrics

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer

from sklearn.metrics import average_precision_score

from sklearn.metrics import r2_score, make_scorer, mean_squared_error

print("scikit-learn libraries imported successfully")



# Other ML algorithms

from lightgbm import LGBMRegressor

print("lightgbm imported")

import xgboost as xgb

print("xgboost imported")

from mlxtend.regressor import StackingCVRegressor, StackingRegressor

print("StackingRegressor imported")
import warnings

warnings.simplefilter('ignore')

#warnings.simplefilter(action='ignore', category=FutureWarning)
# Input data files are available in the "../input/" directory.

print(os.listdir("../input"))

# Any results written to the current directory are saved as output.
# importing the supplied dataset and storing it in a dataframe

training = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# making copies of original datasets for rest of this kernel

df_train = training.copy()

df_test = test.copy()

print(df_train.shape, df_test.shape)
#drop target variable from training dataset

target = df_train['SalePrice']  #target variable

df_train = df_train.drop('SalePrice', axis=1) 



print("Training: {}, Target: {}, Test: {}".format(df_train.shape, target.shape, df_test.shape))
df_train_exp = df_train.copy() #make a copy of the training dataset for EDA purposes

print(df_train_exp.shape) 
df_train_exp.head()
print("{} Numerical columns, {} Categorial columns".format(

    list(df_train_exp.select_dtypes(include=[np.number]).shape)[1],

    list(df_train_exp.select_dtypes(include = ['object']).shape)[1]))
# let's break down the columns by their type (i.e. int64, float64, object)

df_train_exp.columns.to_series().groupby(df_train_exp.dtypes).groups
#list of columns with missing values

print("{} columns have missing values:".format(

    len(df_train_exp.columns[df_train_exp.isna().any()].tolist())))

df_train_exp.columns[df_train_exp.isna().any()].tolist()
df_train_exp.describe() # let's have a look at variable types in our dataframe
df_train_exp.hist(figsize=(18,18))

plt.show()
# Testing for normal distribution hypothesis in numerical features

test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01

numerical_features = [f for f in df_train_exp.columns if df_train_exp.dtypes[f] != 'object']

normal = pd.DataFrame(df_train_exp[numerical_features])

normal = normal.apply(test_normality)

print(not normal.any())
# Calculate correlations

corr = training.corr(method='spearman')

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

# Heatmap

plt.figure(figsize=(15, 10))

sns.heatmap(corr,

            vmax=.5,

            mask=mask,

            #annot=True, 

            fmt='.2f',

            linewidths=.2, cmap="YlGnBu");
# Find correlations with the target and sort

correlations = training.corr(method='spearman')['SalePrice'].sort_values(ascending=False)

correlations_abs = correlations.abs()

print('\nTop 10 correlations (absolute):\n', correlations_abs.head(11))
target_exp = target.copy() #make copy for exploratory purposes
# let's see if there are any missing values (i.e. NA)

print("There are {} NA values in 'SalePrice'".format(target_exp.isnull().values.sum()))
y = target_exp

plt.figure(1); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=stats.lognorm)

plt.ylabel('Frequency')

print("Skewness: %f" % target_exp.skew())

# get mean and standard deviation

(mu, sigma) = norm.fit(target_exp)

print('Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma))
# let's get some stats on the 'SalePrice' variable

print("Statistics for the supplied house prices training dataset:\n")

print("Minimum price: ${:,.2f}".format(np.min(target_exp)))

print("Maximum price: ${:,.2f}".format(np.max(target_exp)))

print("Mean price: ${:,.2f}".format(np.mean(target_exp)))

print("Median price ${:,.2f}".format(np.median(target_exp)))

print("Standard deviation of prices: ${:,.2f}".format(np.std(target_exp)))
#  To get a visual of the outliers, let's plot a box plot.

sns.boxplot(y = target)

plt.ylabel('SalePrice (Log)')

plt.title('Price');



# count number of outliers after transformation is applied

Q1 = target.quantile(0.25)

Q3 = target.quantile(0.75)

IQR = Q3 - Q1

print("IQR value: {}\n# of outliers: {}".format(

    IQR,

    ((target < (Q1 - 1.5 * IQR)) | (target > (Q3 + 1.5 * IQR))).sum()))
#applying log transformation to the Target Variable

target_tr = np.log1p(target)



# let's plot a histogram with the fitted parameters used by the function

sns.distplot(target_tr , fit=norm);

(mu, sigma) = norm.fit(target_tr)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.title('Price (Log)');

print("Skewness: %f" % target_tr.skew())
#  To get a visual of the outliers, let's plot a box plot.

sns.boxplot(y = target_tr)

plt.ylabel('SalePrice (Log)')

plt.title('Price');



# count number of outliers after transformation is applied

Q1 = target_tr.quantile(0.25)

Q3 = target_tr.quantile(0.75)

IQR = Q3 - Q1

print("IQR value: {}\n# of outliers: {}".format(

    IQR,

    ((target_tr < (Q1 - 1.5 * IQR)) | (target_tr > (Q3 + 1.5 * IQR))).sum()))
perc_na = (df_train.isnull().sum()/len(df_train))*100

ratio_na = perc_na.sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Values Ratio' :ratio_na})

print(missing_data.shape)

missing_data.head(20)
def house_pipeline_v1(dataframe,

                      impute_method = "median",

                      feature_transform = "yes",

                      feature_scaling = "RobustScaler", 

                      feature_selection = "yes"):

    # 0. initialising dataframe

    df_pipe = dataframe.copy()

    print("Dataframe loaded.")

    

    # Drop redundant columns

    df_pipe.drop(['Id'], axis=1, inplace=True) # drop Id column

    print("Dropped redundant column 'Id'.")



    # column types variables

    numeric_features = list(df_pipe.select_dtypes(

        include=[np.number]).columns.values)

    categ_features = list(df_pipe.select_dtypes(

        include=['object']).columns.values)

    for col in numeric_features:

        df_pipe[col] = df_pipe[col].astype(float)



    # 1. Handling missing values

    # replacing NaNs in categorical features with "None"

    df_pipe[categ_features] = df_pipe[categ_features].apply(

        lambda x: x.fillna("None"), axis=0)



    # imputing numerical features

    for col in ("LotFrontage", 'GarageYrBlt', 'GarageArea', 'GarageCars'):

        df_pipe[col].fillna(0.0, inplace=True)

        

    if impute_method == "median": # replacing NaNs in numerical features with the median

        df_pipe[numeric_features] = df_pipe[numeric_features].apply(

            lambda x: x.fillna(x.median()), axis=0)

        print("Missing values imputed with median.")

    

    elif impute_method == "mean": # replacing NaNs in numerical features with the mean

        df_pipe[numeric_features] = df_pipe[numeric_features].apply(

            lambda x: x.fillna(x.mean()), axis=0)

        print("Missing values imputed with mean.")



    # 2. Feature Engineering

    # Examples: Discretize Continous Feature;

    #           Decompose Features;

    #           Add Combination of Feature

    df_pipe['YrBltAndRemod']=df_pipe['YearBuilt']+df_pipe['YearRemodAdd']

    df_pipe['TotalSF']=df_pipe['TotalBsmtSF'] + df_pipe['1stFlrSF'] + df_pipe['2ndFlrSF']



    df_pipe['Total_sqr_footage'] = (df_pipe['BsmtFinSF1'] + df_pipe['BsmtFinSF2'] +

                                     df_pipe['1stFlrSF'] + df_pipe['2ndFlrSF'])



    df_pipe['Total_Bathrooms'] = (df_pipe['FullBath'] + (0.5 * df_pipe['HalfBath']) +

                                   df_pipe['BsmtFullBath'] + (0.5 * df_pipe['BsmtHalfBath']))



    df_pipe['Total_porch_sf'] = (df_pipe['OpenPorchSF'] + df_pipe['3SsnPorch'] +

                                  df_pipe['EnclosedPorch'] + df_pipe['ScreenPorch'] + 

                                 df_pipe['WoodDeckSF'])

    print("Feature enginering: added combination of features.")

    

    df_pipe['haspool'] = df_pipe['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    df_pipe['has2ndfloor'] = df_pipe['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    df_pipe['hasgarage'] = df_pipe['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    df_pipe['hasbsmt'] = df_pipe['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    df_pipe['hasfireplace'] = df_pipe['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    print("Feature enginering: added boolean features.")

    

    # 3. Feature Transformations (log(x), sqrt(x), x^2, etc.)

    # Transform numerical features that should be considered as strings 

    df_pipe['MSSubClass'] = df_pipe['MSSubClass'].apply(str)

    df_pipe['YrSold'] = df_pipe['YrSold'].astype(str)

    df_pipe['MoSold'] = df_pipe['MoSold'].astype(str)

    df_pipe['YrBltAndRemod'] = df_pipe['YrBltAndRemod'].astype(str)

    print("Transformed numerical features that should be considered as strings.")

    

    numeric_features = list(df_pipe.select_dtypes(

        include=[np.number]).columns.values)

    categ_features = list(df_pipe.select_dtypes(

        include=['object']).columns.values)

    

    if feature_transform == "yes":

        # Transform all numerical columns with skewness factor > 0.5

        skew_features = df_pipe[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)

        high_skew = skew_features[skew_features > 0.5]

        skew_index = high_skew.index

        for i in skew_index:

            df_pipe[i] = boxcox1p(df_pipe[i], boxcox_normmax(df_pipe[i]+1))

        print("Transformed numerical columns with high skewness factor.")

    elif feature_transform == "no":

        pass



    # 4. Label Encoding

    df_pipe = pd.get_dummies(df_pipe)

    print("Label Encoding: from {} cols to {} cols.".format(

        dataframe.shape[1], df_pipe.shape[1]))



    # 5. Feature Scaling

    #cols = df_pipe.select_dtypes([np.number]).columns

    if feature_scaling == 'MinMaxScaler':

        scaler = MinMaxScaler(feature_range=(0, 1))

        for col in numeric_features:

            df_pipe[[col]] = scaler.fit_transform(df_pipe[[col]])

        print("Performed feature Scaling with MinMaxScaler.")



    elif feature_scaling == 'StandardScaler':

        scaler = StandardScaler()

        for col in numeric_features:

            df_pipe[[col]] = scaler.fit_transform(df_pipe[[col]])

        print("Performed feature Scaling with StandardScaler.")



    elif feature_scaling == "RobustScaler":

        scaler = RobustScaler()

        for col in numeric_features:

            df_pipe[[col]] = scaler.fit_transform(df_pipe[[col]])

        print("Performed feature Scaling with RobustScaler.")

    

    # 6. Feature Selection

    ## let's remove columns with little variance (to reduce overfitting)

    overfit = []

    for i in df_pipe.columns:

        counts = df_pipe[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(df_pipe) * 100 > 99.9: # the threshold is set at 99.9%

            overfit.append(i)

    overfit = list(overfit)

    # let's make sure to keep data processing columns needed later on

    try:

        overfit.remove('Dataset_Train')

        overfit.remove('Dataset_Test')

    except:

        pass

    df_pipe.drop(overfit, axis=1, inplace=True)

    print("To prevent overfitting, {} columns were removed.".format(len(overfit)))

    

    ## Summary

    print("Shape of transformed dataset: {} (original: {})".format(df_pipe.shape, dataframe.shape))

    return df_pipe
def target_transf(target, 

                  transform="log"):

    

    if transform == "log":

        target_tranf = np.log1p(target)

        print("Target feature transformed with natural logarithm.")

    

    elif transform == "sqrt":

        target_tranf = np.sqrt(target)

        print("Target feature transformed with sqrt.")

    

    elif transform == "square":

        target_tranf = np.square(target)

        print("Target feature transformed with square.")

    

    print("Shape of transformed target: {}".format(target_tr.shape))

    return target_tranf
# Test pipeline

df_train_test = house_pipeline_v1(df_train)

print("\n")

target_tr = target_transf(target)
# let's check that we no longer have any missing values

perc_na = (df_train_test.isnull().sum()/len(df_train_test))*100

ratio_na = perc_na.sort_values(ascending=False)

missing_data = pd.DataFrame({'missing_ratio' :ratio_na})

missing_data = missing_data.drop(missing_data[missing_data.missing_ratio == 0].index)

missing_data.head(5)
# target feature transformed

sns.distplot(target_tr , fit=norm);

(mu, sigma) = norm.fit(target_tr)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.title('Price (Log)');

print("Skewness: %f" % target_tr.skew())
## Feature Scaling

col_eda = list(correlations_abs.index)

df_train_scal = df_train_test.filter(col_eda, axis=1).copy()

df_train_scal.hist(figsize=(18,18))

plt.show()
# Copy dataframes prior to data processing

df_train_pipeline = df_train.copy()

df_test_pipeline = df_test.copy()

# Concat dataframes

df_train_pipeline["Dataset"] = "Train"

df_test_pipeline["Dataset"] = "Test"

# Concat dataframes

df_joined = pd.concat([df_train_pipeline, df_test_pipeline], 

                      sort=False)

df_joined = df_joined.reset_index(drop=True) # reset index

print("Joined Dataframe shape: {}".format(df_joined.shape))
df_joined_ml = house_pipeline_v1(df_joined,

                                 impute_method = "median",

                                 feature_transform = "yes",

                                 feature_scaling = "RobustScaler", 

                                 feature_selection = "yes")

print("----\n")

target_ml = target_transf(target)

print("----\n")

print("Transformed Joined Dataframe shape: {}, and target shape: {}".format(

    df_joined_ml.shape, target_ml.shape))
# Extract Training data from joined transformed dataset

df_train_ml = df_joined_ml[df_joined_ml['Dataset_Train']==1].copy()

# Remove redundant features

df_train_ml.drop(['Dataset_Train'], axis=1, inplace=True)

df_train_ml.drop(['Dataset_Test'], axis=1, inplace=True)

# Reset index

df_train_ml = df_train_ml.reset_index(drop=True) 

print(df_train_ml.shape)
# Extract Testing data from joined transformed dataset

df_test_ml = df_joined_ml[df_joined_ml['Dataset_Test']==1].copy()

# Remove redundant features

df_test_ml.drop(['Dataset_Train'], axis=1, inplace=True)

df_test_ml.drop(['Dataset_Test'], axis=1, inplace=True)

# Reset index

df_test_ml = df_test_ml.reset_index(drop=True)

print(df_test_ml.shape)
X_train, X_test, y_train, y_test = train_test_split(df_train_ml,

                                                    target_ml,

                                                    test_size=0.2,

                                                    stratify=df_train_ml['OverallQual'],

                                                    random_state=42)
print("Training Data Shape: {}".format(df_train_ml.shape))

print("X_train Shape: {}".format(X_train.shape))

print("X_test Shape: {}".format(X_test.shape))
# selection of algorithms to consider

models = []

models.append(('Ridge Regression', Ridge(alpha=1.0)))

models.append(('ElasticNet', ElasticNet()))

models.append(('Random Forest', RandomForestRegressor(

    n_estimators=100, random_state=7)))

models.append(('Lasso', Lasso(random_state=42)))

models.append(('XGBoost Regressor', xgb.XGBRegressor(objective='reg:squarederror', 

                                                     random_state=42)))

models.append(('Gradient Boosting Regressor', GradientBoostingRegressor()))

models.append(('LGBM Regressor',LGBMRegressor(objective='regression')))

models.append(('SVR',SVR()))



# set table to table to populate with performance results

rmse_results = []

names = []

col = ['Algorithm', 'RMSE Mean', 'RMSE SD']

df_results = pd.DataFrame(columns=col)



# evaluate each model using cross-validation

kfold = model_selection.KFold(n_splits=5, shuffle = True, random_state=7)

i = 0

for name, model in models:

    # -mse scoring

    cv_mse_results = model_selection.cross_val_score(

        model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

    # calculate and append rmse results

    cv_rmse_results = np.sqrt(-cv_mse_results)

    rmse_results.append(cv_rmse_results)

    names.append(name)

    df_results.loc[i] = [name,

                         round(cv_rmse_results.mean(), 4),

                         round(cv_rmse_results.std(), 4)]

    i += 1

df_results.sort_values(by=['RMSE Mean'], ascending=True).reset_index(drop=True)
fig = plt.figure(figsize=(15, 8))

fig.suptitle('Algorithm RMSE Comparison')

ax = fig.add_subplot(111)

plt.boxplot(rmse_results)

ax.set_xticklabels(names)

plt.show();
import xgboost as xgb

xgb_regressor = xgb.XGBRegressor(random_state=42)
# start = time() # Get start time

# cv_sets_xgb = ShuffleSplit(random_state = 10) # shuffling our data for cross-validation

# parameters_xgb = {'n_estimators':range(2000, 8000, 500), 

#              'learning_rate':[0.05,0.060,0.070], 

#              'max_depth':[3,5,7],

#              'min_child_weight':[1,1.5,2]}

# scorer_xgb = make_scorer(mean_squared_error)

# grid_obj_xgb = RandomizedSearchCV(xgb_regressor, 

#                                  parameters_xgb,

#                                  scoring = scorer_xgb, 

#                                  cv = cv_sets_xgb,

#                                  random_state= 99)

# grid_fit_xgb = grid_obj_xgb.fit(X_train, y_train)

# xgb_opt = grid_fit_xgb.best_estimator_



# end = time() # Get end time

# xgb_time = (end-start)/60 # Calculate training time

# print('It took {0:.2f} minutes for RandomizedSearchCV to converge to optimised parameters for the RandomForest model'.format(xgb_time))

# ## Print results

# print('='*20)

# print("best params: " + str(grid_fit_xgb.best_estimator_))

# print("best params: " + str(grid_fit_xgb.best_params_))

# print('best score:', grid_fit_xgb.best_score_)

# print('='*20)
# XGBoost with tuned parameters

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

xgb_opt = xgb.XGBRegressor(learning_rate=0.01,

                           n_estimators=6000,

                           max_depth=4,

                           min_child_weight=0,

                           gamma=0.6,

                           subsample=0.7,

                           colsample_bytree=0.7,

                           objective='reg:squarederror',

                           nthread=-1,

                           scale_pos_weight=1,

                           seed=27,

                           reg_alpha=0.00006,

                           random_state=42)
gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)
lightgbm = LGBMRegressor(objective='regression', 

                         num_leaves=6,

                         learning_rate=0.01, 

                         n_estimators=7000,

                         max_bin=200, 

                         bagging_fraction=0.8,

                         bagging_freq=4, 

                         bagging_seed=8,

                         feature_fraction=0.2,

                         feature_fraction_seed=8,

                         min_sum_hessian_in_leaf = 11,

                         verbose=-1,

                         random_state=42)
# start = time() # Get start time

# rf_regressor = RandomForestRegressor(random_state=42)

# cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation

# parameters = {'n_estimators':range(5, 950, 5), 

#               'min_samples_leaf':range(20, 40, 5), 

#               'max_depth':range(3, 5, 1)}

# scorer = make_scorer(mean_squared_error)

# n_iter_search = 10

# grid_obj = RandomizedSearchCV(rf_regressor, 

#                               parameters, 

#                               n_iter = n_iter_search, 

#                               scoring = scorer, 

#                               cv = cv_sets,

#                               random_state= 99)

# grid_fit = grid_obj.fit(X_train, y_train)

# rf_opt = grid_fit.best_estimator_

# end = time() # Get end time

# rf_time = (end-start)/60 # Calculate training time

# print('It took {0:.2f} minutes for RandomizedSearchCV to converge to optimised parameters for the RandomForest model'.format(rf_time))

# ## Print results

# print('='*20)

# print("best params: " + str(grid_fit.best_estimator_))

# print("best params: " + str(grid_fit.best_params_))

# print('best score:', grid_fit.best_score_)

# print('='*20)
# RandomForest with tuned parameters

rf_reg = RandomForestRegressor(n_estimators=100, 

                               random_state=7)

rf_opt = RandomForestRegressor(n_estimators=1200,

                               max_depth=15,

                               min_samples_split=5,

                               min_samples_leaf=5,

                               max_features=None,

                               oob_score=True,

                               random_state=42)
rf_imp = RandomForestRegressor(n_estimators=1200,

                               max_depth=15,

                               min_samples_split=5,

                               min_samples_leaf=5,

                               max_features=None,

                               oob_score=True,

                               random_state=42)

rf_imp.fit(X_train, y_train)

importances = rf_imp.feature_importances_

df_param_coeff = pd.DataFrame(columns=['Feature', 'Coefficient'])

for i in range(len(X_train.columns)-1):

    feat = X_train.columns[i]

    coeff = importances[i]

    df_param_coeff.loc[i] = (feat, coeff)

df_param_coeff.sort_values(by='Coefficient', ascending=False, inplace=True)

df_param_coeff = df_param_coeff.reset_index(drop=True)

print("Top 10 features:\n{}".format(df_param_coeff.head(10)))



importances = rf_imp.feature_importances_

indices = np.argsort(importances)[::-1] # Sort feature importances in descending order

names = [X_train.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances

plt.figure(figsize=(15, 7)) # Create plot

plt.title("Top 10 Most Important Features") # Create plot title

plt.bar(range(10), importances[indices][:10]) # Add bars

plt.xticks(range(10), names[:10], rotation=90) # Add feature names as x-axis labels

#plt.bar(range(X_train.shape[1]), importances[indices]) # Add bars

#plt.xticks(range(X_train.shape[1]), names, rotation=90) # Add feature names as x-axis labels

plt.show() # Show plot
kfolds = KFold(n_splits=5, shuffle=True, random_state=7)

rcv_alphas = np.arange(14, 16, 0.1)

ridge = RidgeCV(alphas=rcv_alphas, 

                cv=kfolds)
svr = SVR(C= 20, 

          epsilon= 0.008, 

          gamma=0.0003)
# selection of algorithms to consider

start = time() # Get start time

models = []

models.append(('Ridge Regression', ridge))

models.append(('Random Forest', rf_opt))

models.append(('XGBoost Regressor', xgb_opt))

models.append(('Gradient Boosting Regressor', gbr))

models.append(('LGBM Regressor',lightgbm))

models.append(('SVR',svr))

models.append(('StackingRegressor',StackingRegressor(regressors=(gbr,

                                                                 xgb_opt,

                                                                 lightgbm,

                                                                 rf_opt,

                                                                 ridge, 

                                                                 svr),

                                                     meta_regressor=xgb_opt,

                                                     use_features_in_secondary=False)))



# set table to table to populate with performance results

rmse_results = []

names = []

col = ['Algorithm', 'RMSE Mean', 'RMSE SD']

df_results = pd.DataFrame(columns=col)



# evaluate each model using cross-validation

kfold = model_selection.KFold(n_splits=5, shuffle = True, random_state=7)

i = 0

for name, model in models:

    print("Evaluating {}...".format(name))

    # -mse scoring

    cv_mse_results = model_selection.cross_val_score(

        model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

    # calculate and append rmse results

    cv_rmse_results = np.sqrt(-cv_mse_results)

    rmse_results.append(cv_rmse_results)

    names.append(name)

    df_results.loc[i] = [name,

                         round(cv_rmse_results.mean(), 4),

                         round(cv_rmse_results.std(), 4)]

    i += 1

end = time() # Get end time

eval_time = (end-start)/60 # Calculate training time

print('Evaluation completed.\nIt took {0:.2f} minutes to evaluate all models using a 5-fold cross-validation.'.format(eval_time))

df_results.sort_values(by=['RMSE Mean'], ascending=True).reset_index(drop=True)
fig = plt.figure(figsize=(20, 8))

fig.suptitle('Algorithm RMSE Comparison')

ax = fig.add_subplot(111)

plt.boxplot(rmse_results)

ax.set_xticklabels(names)

plt.show()
stack_gen = StackingCVRegressor(regressors=(gbr,

                                            xgb_opt,

                                            lightgbm,

                                            rf_opt,

                                            ridge, 

                                            svr),

                                meta_regressor=xgb_opt,

                                use_features_in_secondary=False)
print('Fitting models to the training data:')

start = time() # Get start time



print('xgboost....')

xgb_model_full_data = xgb_opt.fit(df_train_ml, target_ml)

print('GradientBoosting....')

gbr_model_full_data = gbr.fit(df_train_ml, target_ml)

print('lightgbm....')

lgb_model_full_data = lightgbm.fit(df_train_ml, target_ml)

print('RandomForest....')

rf_model_full_data = rf_opt.fit(df_train_ml, target_ml)

print('Ridge....')

ridge_model_full_data = ridge.fit(df_train_ml, target_ml)

print('SVR....')

svr_model_full_data = svr.fit(df_train_ml, target_ml)

print('Stacking Regression....')

stack_gen_model = stack_gen.fit(np.array(df_train_ml), np.array(target_ml))



end = time() # Get end time

fitting_time = (end-start)/60 # Calculate training time

print('Fitting completed.\nIt took {0:.2f} minutes to fit all the models to the training data.'.format(fitting_time))
def blend_models_predict(X):

    return ((0.25 * stack_gen_model.predict(np.array(X))) + \

            (0.25 * gbr_model_full_data.predict(X)) + \

            (0.15 * svr_model_full_data.predict(X)) + \

            (0.15 * lgb_model_full_data.predict(X)) + \

            (0.1 * ridge_model_full_data.predict(X))+ \

            (0.05 * xgb_model_full_data.predict(X)) + \

            (0.05 * rf_model_full_data.predict(X)) 

           )
# Generate predictions from the blend

y_pred_final = np.floor(np.expm1(blend_models_predict(df_test_ml)))
# Generate submission dataframe

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred_final})



# Exporting submission to CSV

my_submission.to_csv('submission-080719_v1.csv', index=False)