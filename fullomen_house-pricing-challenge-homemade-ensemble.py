# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns



from scipy.stats import norm

from scipy import stats

from math import log, sqrt, exp



from sklearn.pipeline import Pipeline

from sklearn.linear_model import Lasso, LinearRegression, ElasticNet, ElasticNetCV, BayesianRidge

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV

from sklearn.preprocessing import RobustScaler, MaxAbsScaler

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, AdaBoostRegressor

from sklearn import metrics

from sklearn.metrics import r2_score



from functools import reduce



from time import time



from xgboost import XGBRegressor



base = "./kaggle_data/"
def sanitize_na(df, median = None, mode = None):

    

    df_na = df.isnull().sum()

    print("Features with missing values before: ", df_na.drop(df_na[df_na == 0].index))

    

    # Using data description, fill these missing values with "None"

    for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",

               "GarageType", "GarageFinish", "GarageQual", "GarageCond",

               "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",

                "BsmtFinType2", "MSSubClass", "MasVnrType"):

        df[col].fillna('None',  inplace=True)



    if median is None:

        # The area of the lot out front is likely to be similar to the houses in the local neighbourhood

        # Therefore, let's use the median value of the houses in the neighbourhood to fill this feature (median because it is resistant to outliers)

        median = df.groupby("Neighborhood")["LotFrontage"].median()

        

    for k, v in median.items():

         df.loc[df["Neighborhood"] == k,"LotFrontage"] = df[df["Neighborhood"] == k]["LotFrontage"].fillna(v)    

    



    # Using data description, fill these missing values with 0 

    for col in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", 

               "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",

               "BsmtFullBath", "BsmtHalfBath"):

        df[col].fillna(0, inplace=True)

        

    # Fill these features with their mode, the most commonly occuring value. This is okay since there are a low number of missing values for these features

    modecols = ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType','Functional']

    if mode is None:

        mode = {}

        for col in modecols:

            mode[col] = df[col].mode()[0]

    for col in modecols:

        df[col].fillna(mode[col], inplace=True)

        



    df_na = df.isnull().sum()

    print("Features with missing values after: ", df_na.drop(df_na[df_na == 0].index))

    return median, mode



def map_ordinal_features(df):

    # External quality and condition

    exter_qual_todict = {'Fa':1, 'TA':2, 'Gd':3, 'Ex':4, 'None':0}

    df['ExterQual']=df.ExterQual.apply(exter_qual_todict.get)



    exter_cond_todict =  {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'None':0}

    df['ExterCond']=df.ExterCond.apply(exter_cond_todict.get)



    # Land slope

    land_slope_todict = {'Gtl':1, 'Mod':2, 'Sev':3}

    df['LandSlope']=df.LandSlope.apply(land_slope_todict.get)



    # Basement

    bsmt_qual_todict = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'None':0}

    df['BsmtQual']=df.BsmtQual.apply(bsmt_qual_todict.get)



    bsmt_cond_todict = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'None':0}

    df['BsmtCond'] = df.BsmtCond.apply(bsmt_cond_todict.get)



    bsmt_exposure_todict = {'No':1, 'Mn':2, 'Av':3, 'Gd':4, 'None':0}

    df['BsmtExposure'] = df.BsmtExposure.apply(bsmt_exposure_todict.get)



    bsmt_fin_type_todict = {'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}

    df['BsmtFinType1'] = df.BsmtFinType1.apply(bsmt_fin_type_todict.get)

    df['BsmtFinType2'] = df['BsmtFinType2'].apply(bsmt_fin_type_todict.get)



    # Heating

    heating_qc_todict = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}

    df['HeatingQC'] = df['HeatingQC'].apply(heating_qc_todict.get)



    # Kitchen

    kitchen_qual_todict = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}

    df['KitchenQual'] = df['KitchenQual'].apply(kitchen_qual_todict.get)



    # Fireplace

    fireplace_qu_todict = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'None':0}

    df['FireplaceQu'] = df['FireplaceQu'].apply(fireplace_qu_todict.get)



    # Garage

    garage_finish_todict = {'None':0, 'Unf':1, 'RFn':2, 'Fin':3}

    df['GarageFinish'] = df['GarageFinish'].apply(garage_finish_todict.get)



    garage_qual_todict = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'None':0}

    df['GarageQual'] = df['GarageQual'].apply(garage_qual_todict.get)



    garage_cond_todict = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'None':0}

    df['GarageCond'] = df['GarageCond'].apply(garage_cond_todict.get)



    # Paved Driveway

    paved_drive_todict = {'N':0, 'P':1, 'Y':2}

    df['PavedDrive'] = df['PavedDrive'].apply(paved_drive_todict.get)
# Read the .csv file 

rawTrainDF = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
print('Train dataset size: ', rawTrainDF.Id.size)

print('Test dataset size: ', test_df.Id.size)
# aggregate all null values 

rawTrainDF_na = rawTrainDF.isnull().sum()



# get rid of all the values with 0 missing values and plot 

rawTrainDF_na = rawTrainDF_na.drop(rawTrainDF_na[rawTrainDF_na == 0].index).sort_values(ascending=False)

plt.subplots(figsize =(10, 7))

rawTrainDF_na.plot(kind='bar');
print('Cleaning training data...')

median, mode = sanitize_na(rawTrainDF);
fig, ax = plt.subplots(ncols=3, figsize=(24,5))



sns.scatterplot(rawTrainDF.Id, rawTrainDF.SalePrice, palette='ocean', ax=ax[0])

ax[0].set_title('Sale Prices')



sns.distplot(rawTrainDF['SalePrice'] , fit=norm, ax=ax[1]);



(mu, sigma) = norm.fit(rawTrainDF['SalePrice']) # parameters for an ideal gaussian distribution with our data

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

ax[1].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

ax[1].set_ylabel('Frequency')

ax[1].set_title('SalePrice distribution')



res = stats.probplot(rawTrainDF['SalePrice'], plot=plt)

plt.show()



print("Skewness: %f" % rawTrainDF['SalePrice'].skew())

print("Kurtosis: %f" % rawTrainDF['SalePrice'].kurt())
fig, ax = plt.subplots(ncols=2, figsize=(15,5))



rawTrainDF['SalePrice'].hist(ax=ax[0], cumulative=True, density=True, bins=1000, histtype='step')



colors = ['r', 'y', 'g', 'violet', 'orange']

# Calculate and plot the 5 important percentiles

for i, quantile in enumerate([.1, 0.25, .5, .75, .9]):

    ax[0].axvline(rawTrainDF['SalePrice'].quantile(quantile), label='%.0fth perc.' % (quantile*100), color=colors[i])



ax[0].set_title('ECDF of Sale Price per HouseID')

ax[0].set_xlabel('Sale Price')

ax[0].set_ylabel('ECDF')

ax[0].legend(loc='best')



cleanedTrainDF = rawTrainDF[rawTrainDF['SalePrice'] <= 400000]



cleanedTrainDF['SalePrice'].hist(ax=ax[1], cumulative=True, density=True, bins=1000, histtype='step')

ax[1].set_title('ECDF of Sale Price per HouseID')

ax[1].set_xlabel('Sale Price')

ax[1].set_ylabel('ECDF')



plt.show()
fig, ax = plt.subplots(ncols = 2, figsize=(10,4))



sns.barplot(x=cleanedTrainDF.Utilities.unique(), y=cleanedTrainDF.groupby(by='Utilities', as_index=False).count()['Id'], palette="rocket", ax=ax[0])

ax[0].set_ylabel('Count')

ax[0].set_xlabel('Utilities')



sns.barplot(x=cleanedTrainDF.Street.unique(), y=cleanedTrainDF.groupby(by='Street', as_index=False).count()['Id'], palette="rocket", ax=ax[1])

ax[1].set_ylabel('Count')

ax[1].set_xlabel('Street')



print("Distinct values for column 'Utilities': ", cleanedTrainDF.Utilities.unique())

print("Number of times 'NoSeWa' appears in the dataset:", cleanedTrainDF.Utilities[cleanedTrainDF.Utilities == 'NoSeWa'].size)
columns_todrop = ['Utilities', 'PoolArea', 'PoolQC', 'Street', 'Functional', 'RoofMatl']

cleanedTrainDF = cleanedTrainDF.drop(columns_todrop, axis=1)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

sns.barplot(x='BsmtQual', y='SalePrice', data=cleanedTrainDF, palette="rocket", ax=ax[0][0])

sns.boxplot(x='BsmtQual', y='SalePrice', data=cleanedTrainDF, palette='rocket', ax=ax[0][1])

sns.countplot(x='BsmtQual', data=cleanedTrainDF, ax=ax[0][2])

ax[0][2].set_ylabel('Count')

ax[0][2].set_xlabel('BsmtQual')





sns.barplot(x='ExterQual', y='SalePrice', data=cleanedTrainDF, palette="rocket", ax=ax[1][0])

sns.boxplot(x='ExterQual', y='SalePrice', data=cleanedTrainDF, palette='rocket', ax=ax[1][1])

sns.countplot(x='ExterQual', data=cleanedTrainDF, ax=ax[1][2])

ax[1][2].set_ylabel('Count')

ax[1][2].set_xlabel('ExterQual')



plt.show()
map_ordinal_features(cleanedTrainDF)
fig, ax = plt.subplots(ncols=2, figsize=(15,5))

sns.boxplot(x='Foundation', y='SalePrice', data=cleanedTrainDF, ax=ax[0])

sns.countplot(x='Foundation', data=cleanedTrainDF, ax=ax[1])

plt.show()
# if Foundation doesn't belong to PConc or CBlock, convert it to 'Other'

cleanedTrainDF.loc[cleanedTrainDF['Foundation'].isin(['BrkTil', 'Wood', 'Slab', 'Stone']),'Foundation'] = 'Other'



fig, ax = plt.subplots(ncols=3, figsize=(20,5))

sns.barplot(x='Foundation', y='SalePrice', data=cleanedTrainDF, ax=ax[0])

sns.boxplot(x='Foundation', y='SalePrice', data=cleanedTrainDF, ax=ax[1])

sns.countplot(x='Foundation', data=cleanedTrainDF, ax=ax[2])

plt.show()
foundation_todict = {'PConc':3, 'CBlock':2, 'Other':1}

cleanedTrainDF['Foundation'] = cleanedTrainDF['Foundation'].apply(foundation_todict.get)
fig, ax = plt.subplots(ncols=3, figsize=(20,5))

sns.boxplot(x='Fence', y='SalePrice', data=cleanedTrainDF, ax=ax[0])

sns.countplot(x='Fence', data=cleanedTrainDF, ax=ax[1])

sns.scatterplot(x='Fence', y='SalePrice', data=cleanedTrainDF, ax=ax[2])

plt.show()
cleanedTrainDF = cleanedTrainDF.drop('Fence', axis=1)
def diff_remod_built(df):

    df['RemodDiffYear'] =  - (df['YrSold'] - df['YearRemodAdd'])

    df['RemodDiffYear2'] = - (2010 - df['YearRemodAdd'])

    df = df.drop('YearRemodAdd', axis=1)

    return df
cleanedTrainDF = diff_remod_built(cleanedTrainDF)
# bins has to be a sequence of scalars

def year_binner(df, binsYearBuilt, binsGarageBuilt):

    df['YearBuilt'] = pd.cut(df['YearBuilt'], binsYearBuilt, labels=False)

    lab = []

    pre = None

    for binz in binsGarageBuilt:

        if pre is None:

            pre = str(binz)

            continue

        lab.append(pre+"-"+str(binz))

        pre=str(binz)

    df['GarageYrBlt'] = pd.cut(df['GarageYrBlt'], binsGarageBuilt, labels=lab)
fig, ax = plt.subplots(ncols=2, figsize=(25, 5))

sns.scatterplot(x='YearBuilt', y='SalePrice', data=cleanedTrainDF, ax=ax[0])



# get equi-depth bins from YearBuilt

year_built_bins = pd.qcut(cleanedTrainDF['YearBuilt'], 6)

sns.boxplot(x=year_built_bins, y=cleanedTrainDF['SalePrice'], ax=ax[1])

plt.show()



print('YearBuilt bins: ', year_built_bins.unique())
fig, ax = plt.subplots(ncols=2, figsize=(25, 5))

sns.scatterplot(x='GarageYrBlt', y='SalePrice', data=cleanedTrainDF[cleanedTrainDF['GarageYrBlt']>0], ax=ax[0])

gryear_built_bins = pd.qcut(cleanedTrainDF.loc[cleanedTrainDF['GarageYrBlt']>1,'GarageYrBlt'], 6)

sns.boxplot(x=gryear_built_bins, y=cleanedTrainDF.loc[cleanedTrainDF['GarageYrBlt']>0,'SalePrice'], ax=ax[1])

plt.show()



print('GarageYrBlt bins: ', gryear_built_bins.unique())
# transform YearBuilt in an ordinal feature based on bucket representing range of years

yrbins = [1,1940,1959,1972,1994,2004,2010]

gybins = [-1, 1, 1954, 1967, 1980, 1997, 2004, 2010]

year_binner(cleanedTrainDF, yrbins, gybins)
def yrsold_scaleback(df, year_zero):

    df["YrSold"] = df["YrSold"] - year_zero
yrsold_scaleback(cleanedTrainDF, 2006)
def test_cleaning_pipeline(df, columns_todrop, median, mode, yrbins, gybins, foundation_todict):

    sanitize_na(df, median, mode);

    df = df.drop(columns_todrop, axis=1)

    map_ordinal_features(df)

    df = df.drop('Fence', axis=1)

    df = diff_remod_built(df)

    year_binner(df, yrbins, gybins)

    yrsold_scaleback(df, 2006)

    df.drop("Id", axis = 1, inplace = True)

    df.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)

    test_df.loc[test_df['Foundation'].isin(['BrkTil', 'Wood', 'Slab', 'Stone']),'Foundation'] = 'Other'

    test_df['Foundation'] = test_df['Foundation'].apply(foundation_todict.get)

    return df
#Save the 'Id' column

train_ID = cleanedTrainDF['Id']

train_price = cleanedTrainDF['SalePrice']

test_ID = test_df['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train_df = cleanedTrainDF.drop("Id", axis = 1)

train_df.drop('SalePrice', axis=1, inplace=True)

train_df.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)
test_df = test_cleaning_pipeline(test_df, columns_todrop, median, mode, yrbins, gybins, foundation_todict)
assert(test_df.keys().size == train_df.keys().size)



train_dummies = pd.get_dummies(train_df)

test_dummies = pd.get_dummies(test_df)



train_dummies, test_dummies = train_dummies.align(test_dummies, join='outer', axis=1, fill_value=0)



print(train_dummies.keys().size)

print(test_dummies.keys().size)



assert(test_dummies.keys().size == train_dummies.keys().size)



train_dummies['SalePrice'] = train_price
#correlation matrix

corrmat = rawTrainDF.corr()

f, ax = plt.subplots(figsize=(15, 15))

k = 21 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(rawTrainDF[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
X_train, X_valid, y_train, y_valid = train_test_split(train_dummies[train_dummies.keys()[:-1]],  train_dummies['SalePrice'], test_size=0.2)

# Log Trasformation

y_train, y_valid = [log(y) for y in y_train],  [log(y) for y in y_valid]



X = [X_train, X_valid]

Y = [y_train, y_valid]

for x in X:

    print("Shape of dataframe x: ", x.shape)
# Return the cardinality of the grid

def get_cardinality_of_exploration(d):

    return reduce(lambda a, b: a * b, [len(i) for i in d.values()])





# This function returns the value of the metrix utilized to calculate the error

def log_err(target, prediction):

    return sqrt(mean_squared_error(target, prediction))





# This function allows to optimize the accuracy of the model across different parameters configurations generated by the grid

def OptimiseParameters(model, X, y, parameters_grid, n_jobs=1, scoring='r2', instantiated=False):

    # with this option is possible to pass an unistantiated model to the function

    if not instantiated:

        modelCV = model()

    # The GridSearchCV class allows to explore all the parameters configurations generated by the grid

    cross_validation_model = GridSearchCV(estimator=modelCV, param_grid=parameters_grid, cv=10, n_jobs=n_jobs, scoring=scoring)

    cross_validation_model.fit(X,y)

    print("Best parameters configuration: {}\n".format(cross_validation_model.best_params_))

    print("Best R2 score: {}\n".format(cross_validation_model.best_score_))

    # best model is returned

    return cross_validation_model





# This functions is just an utility to display the error of all the models within a dictionary.

def show_errors(d, X, y_true):

    print("ERRORS: \n")

    for k, v in d.items():

        print("Regressor: {0}".format(k))

        y_pred = v.predict(X)

        print("Error on testing dataset: {0:.3f}\n".format(log_err(y_pred, y_true)))

    print("\n\n")





def average_prediction(regressors, weights, data):

    if len(regressors) != len(weights):

        return None

    return reduce(lambda a,b : a + b, [w * pred.predict(data) for w, pred in zip(weights, regressors)])
# Instantiate an empty dictionary, here all the models will be stored

d = {}
#ELASTIC NET

ElNet_model = ElasticNetCV(l1_ratio=0.03,

                           alphas=[np.arange(0.1, 1, 0.1)],

                           max_iter=1200)



elnetcv_model = ElNet_model

# elnetcv_model = Pipeline([    

#         ('gb', ElNet_model)

#  ])



elnetcv_model.fit(X_train, y_train)



d['ElasticNetCV'] = elnetcv_model
#GBOOST REGRESSOR



GBoost_model = GradientBoostingRegressor(loss="ls",

                                         learning_rate=0.02,

                                         n_estimators=400,

                                        max_depth=3,

                                        alpha=0.05)



gboost_model = GBoost_model



gboost_model.fit(X_train, y_train)



d['GBoost'] = GBoost_model
# XGBOOST REGRESSOR

xgboost_model = XGBRegressor(reg_alpha=0.02, reg_lambda=0.3, n_estimators=2000, gamma=0.0005, max_depth=2, eta=0.05)



xgboost_model.fit(X_train, y_train)



d['XGBoost'] = xgboost_model
#ADABOOST REGRESSOR

adaBoost_model = AdaBoostRegressor(n_estimators=350,

                                   learning_rate=0.07,

                                   loss="square")



adaBoost_model.fit(X_train, y_train)





d['ADABoost'] = adaBoost_model
# LASSO REGRESSOR



lassoModel = Lasso(alpha=0.0005, random_state=2)



lasso_model = lassoModel



lasso_model.fit(X_train, y_train)



d['LASSO'] = lasso_model
bayesian_ridge = BayesianRidge(compute_score=True)



bayesian_ridge.fit(X_train, y_train)



d['BayesianRidge'] = bayesian_ridge
# XGBOOST REGRESSOR



now = time()



# lambda, alpha, eta, gamma

parameters_grid = {

    'lambda' : [0.3],

    'alpha': [0.05],

    'eta': [0.05],

    'max_depth' : [4, 5],

    'n_estimators': [2800, 3500],

    'gamma': [0, 0.005]

}



print("[XGBOOST] Cardinatilty of grid: ", get_cardinality_of_exploration(parameters_grid))

xgboost_model_cv = OptimiseParameters(XGBRegressor, X_train, y_train, parameters_grid=parameters_grid, n_jobs=4)   

after = time()

print("[XGBOOST] parameters, time: ", after - now)



d['best_XGBoost'] = xgboost_model_cv.best_estimator_
# BAYESIAN RIDGE



now = time()



# lambda, alpha, eta, gamma

parameters_grid = {

    'normalize' : [True, False],

    'n_iter' : [400, 500, 700]

}



print("[Bayesian Ridge] Cardinatilty of grid: ", get_cardinality_of_exploration(parameters_grid))

bayesian_ridge_model_cv = OptimiseParameters(BayesianRidge, X_train, y_train, parameters_grid=parameters_grid, n_jobs=4)   

after = time()

print("[Bayesian Ridge] parameters, time: ", after - now)



d['best_BayesianRidge'] = bayesian_ridge_model_cv.best_estimator_
## Gradient Boosting

now = time()



parameters_grid = {

    'loss' : ["huber"],

    'learning_rate':[0.05],

    'min_samples_leaf': [10],

    'n_estimators': [500],

    'max_depth':[3],

    'alpha': [0.3],

    'max_features':['sqrt']

}



print("[Gradient Boost] Cardinatilty of grid: ", get_cardinality_of_exploration(parameters_grid))

best_gbr_model = OptimiseParameters(GradientBoostingRegressor,X_train,y_train,parameters_grid=parameters_grid, n_jobs=4)       

after = time()

print("[Gradient Boost] time spent to get the solution: ", after - now)

d['Best_GradientBoost'] = best_gbr_model.best_estimator_
#ELASTIC NET

now = time()



parameters_grid = {

    'l1_ratio' : [0.001],

    'alphas': [np.arange(0.1, 2, 0.1)],

    'max_iter': [300]

}



print("[ElasticNetCV] Cardinatilty of grid: ", get_cardinality_of_exploration(parameters_grid))

best_encv_model = OptimiseParameters(ElasticNetCV, X_train,y_train,parameters_grid=parameters_grid, n_jobs=4)  

after = time()

print("[ElasticNetCV] time spent to get the solution:", after - now)



d['Best_ElasticNetCV'] = best_encv_model.best_estimator_
show_errors(d, X_valid, y_valid)
selected_regressors = [d['best_XGBoost'], d['Best_GradientBoost'], d['best_BayesianRidge']]

# selected_regressors = [d['BayesianRidge'], d['XGBoost'], d['GBoost']]



weights = [0.4, 0.4, 0.2]



predictions = average_prediction(selected_regressors, weights, X_valid)



print(log_err(predictions, y_valid))
test_predictions = average_prediction(selected_regressors, weights, test_dummies)



# SUBMISSION

submission = pd.DataFrame(test_ID)



exp_pred = [exp(p) for p in test_predictions]



submission['SalePrice'] = exp_pred

submission.to_csv('../submission.csv', index=False)
submission.to_csv('./submission.csv', index=False)
submission