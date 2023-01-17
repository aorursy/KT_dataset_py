import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')

plt.style.use('seaborn-deep')

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.serif'] = 'Ubuntu'

plt.rcParams['font.monospace'] = 'Ubuntu Mono'

plt.rcParams['font.size'] = 10

plt.rcParams['axes.labelsize'] = 12

plt.rcParams['axes.titlesize'] = 12

plt.rcParams['xtick.labelsize'] = 8

plt.rcParams['ytick.labelsize'] = 8

plt.rcParams['legend.fontsize'] = 12

plt.rcParams['figure.titlesize'] = 14

plt.rcParams['figure.figsize'] = (12, 8)



pd.options.mode.chained_assignment = None

pd.options.display.float_format = '{:.2f}'.format

pd.set_option('display.max_columns', 200)

pd.set_option('display.width', 400)

import warnings

warnings.filterwarnings('ignore')

import sklearn.metrics as skm

import sklearn.model_selection as skms

import sklearn.preprocessing as skp

import random

seed = 12

np.random.seed(seed)



from datetime import date
# important funtions

def datasetShape(df):

    rows, cols = df.shape

    print("The dataframe has",rows,"rows and",cols,"columns.")

    

# select numerical and categorical features

def divideFeatures(df):

    numerical_features = housing.select_dtypes(include=[np.number]).drop('SalePrice', axis=1)

    categorical_features = housing.select_dtypes(include=[np.object])

    return numerical_features, categorical_features
data_file = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"

housing = pd.read_csv(data_file)

housing.head()
# check dataset shape

datasetShape(housing)



# check for duplicates

if(len(housing) == len(housing.Id.unique())):

    print("No duplicates found!!")

else:

    print("Duplicates occuring")
numerical_features, categorical_features = divideFeatures(housing)

housing.head()
# boxplots of numerical features for outlier detection



fig = plt.figure(figsize=(16,30))

for i in range(len(numerical_features.columns)):

    fig.add_subplot(9, 5, i+1)

    sns.boxplot(y=numerical_features.iloc[:,i])

plt.tight_layout()

plt.show()
# distplots for categorical data



fig = plt.figure(figsize=(16,30))

for i in range(len(categorical_features.columns)):

    fig.add_subplot(9, 5, i+1)

    categorical_features.iloc[:,i].hist()

    plt.xlabel(categorical_features.columns[i])

plt.tight_layout()

plt.show()
# plotting numerical features for bar plot patterns



discrete_features=[feature for feature in list(numerical_features.columns) if numerical_features[feature].unique().shape[0]<25 and feature not in ['Id', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']]

fig = plt.figure(figsize=(16,30))

for i,feature in enumerate(discrete_features):

    fig.add_subplot(6, 3, i+1)

    data=housing.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

plt.tight_layout()

plt.show()
# plot missing values



def calc_missing(df):

    missing = df.isna().sum().sort_values(ascending=False)

    missing = missing[missing != 0]

    missing_perc = missing/df.shape[0]*100

    return missing, missing_perc



missing, missing_perc = calc_missing(housing)

missing.plot(kind='bar',figsize=(16,6))

plt.title('Missing Values')

plt.show()
import matplotlib.gridspec as gridspec

fig = plt.figure(constrained_layout=True, figsize=(16,6))

grid = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

ax1 = fig.add_subplot(grid[0, :2])

ax1.set_title('Histogram')

sns.distplot(housing.loc[:,'SalePrice'], norm_hist=True, ax = ax1)

ax3 = fig.add_subplot(grid[:, 2])

ax3.set_title('Box Plot')

sns.boxplot(housing.loc[:,'SalePrice'], orient='v', ax = ax3)

plt.show()
# scatterplot for correlation analysis of features with SalePrice



fig = plt.figure(figsize=(16,30))

for i in range(len(numerical_features.columns)):

    fig.add_subplot(8, 5, i+1)

    sns.regplot(x=housing['SalePrice'],y=numerical_features.iloc[:,i])

plt.tight_layout()

plt.show()
# boxplot for distribution analysis of categorical features with SalePrice



fig = plt.figure(figsize=(20,50))

for i in range(len(categorical_features.columns)):

    fig.add_subplot(11, 4, i+1)

    sns.boxplot(y=housing['SalePrice'],x=categorical_features.iloc[:,i])

plt.tight_layout()

plt.show()
# correlation heatmap for all features



plt.figure(figsize = (30,20))

mask = np.zeros_like(housing.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(housing.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0)

plt.show()
# drop the Id

housing.drop('Id', axis=1, inplace=True)



# drop the duplicate rows

housing.drop_duplicates(inplace=True)

datasetShape(housing)
# remove all columns having no values

housing.dropna(axis=1, how="all", inplace=True)

housing.dropna(axis=0, how="all", inplace=True)

datasetShape(housing)
# remove columns having null values more than 30%

housing.dropna(thresh=housing.shape[0]*0.7,how='all',axis=1, inplace=True)

datasetShape(housing)
print("Showing all features:")

print(list(housing.columns))
featuresToDrop = ['MiscVal', 'Exterior2nd']

housing.drop(featuresToDrop, axis=1, inplace=True)

datasetShape(housing)
# missing values with percentage



missing, missing_perc = calc_missing(housing)

pd.concat([missing, missing_perc], axis=1, keys=['Total','Percent'])
housing[missing.index].describe()
housing.loc[housing.LotFrontage.isna(), 'LotFrontage'] = housing.LotFrontage.median()

print("Missing values in LotFrontage:",housing.LotFrontage.isna().sum())
housing.loc[housing.GarageType.isnull(), 'GarageType'] = 'NoGarage'

housing.loc[housing.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'

housing.loc[housing.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'

housing.loc[housing.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'
housing.loc[housing.GarageYrBlt.isna(), 'GarageYrBlt'] = housing.GarageYrBlt.median()

print("Missing values in GarageYrBlt:",housing.GarageYrBlt.isna().sum())
housing.loc[housing.MasVnrType.isnull(), 'MasVnrType'] = 'None'

housing.loc[housing.MasVnrType == 'None', 'MasVnrArea'] = 0

print("Missing values in MasVnrType:",housing.MasVnrType.isna().sum())

print("Missing values in MasVnrArea:",housing.MasVnrArea.isna().sum())
housing.loc[housing.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'

housing.loc[housing.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'

housing.loc[housing.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'

housing.loc[housing.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'

housing.loc[housing.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'
housing.loc[housing['Electrical'].isnull(), 'Electrical'] = 'SBrkr'

missing, missing_perc = calc_missing(housing)

print("Any Missing Values?",missing.values)
# plot sample skewed feature

plt.figure(figsize=(10,4))

sns.distplot(housing.SalePrice)

plt.show()
# extract all skewed features

temp_numerical_features, temp_categorical_features = divideFeatures(housing)

temp_numerical_features['SalePrice'] = housing.SalePrice

# remove categorical features stored as int

temp_numerical_features.drop(['OverallCond', 'OverallQual'], axis=1, inplace=True)

skewed_features = temp_numerical_features.apply(lambda x: x.skew()).sort_values(ascending=False)
# transform skewed features

for feat in skewed_features.index:

    if skewed_features.loc[feat] > 0.5:

        housing[feat] = np.log1p(housing[feat])
# plot sample treated feature

plt.figure(figsize=(10,4))

sns.distplot(housing.SalePrice)

plt.show()
# outlier treatment for categorical features

def getCategoricalSkewed(categories, threshold):

    tempSkewedFeatures = []

    for feat in categories:

        for featValuePerc in list(housing[feat].value_counts()/housing.shape[0]):

            if featValuePerc > threshold:

                tempSkewedFeatures.append(feat)

    return list(set(tempSkewedFeatures))



# display all categorical skewed features which have value_counts > 90%

categoricalSkewed = getCategoricalSkewed(temp_categorical_features.columns, .90)

for feat in categoricalSkewed:

    print(housing[feat].value_counts()/len(housing))

    print()
print("Before Removing:")

datasetShape(housing)



# removing skewed categorical data 

housing.drop(categoricalSkewed, axis=1, inplace=True)

print("After Removing:")

datasetShape(housing)
numerical_features, categorical_features = divideFeatures(housing)

categorical_features.columns
# housing overallqual in four bins

housing['OverallQual'].replace([1,2,3,4,5,6,7,8,9,10], ['Poor', 'Poor', 'Fair', 'Average', 'Average', 'Good', 'Good', 'Excellent', 'Excellent', 'Excellent'], inplace=True)

housing['OverallCond'].replace([1,2,3,4,5,6,7,8,9,10], ['Poor', 'Poor', 'Fair', 'Average', 'Average', 'Good', 'Good', 'Excellent', 'Excellent', 'Excellent'], inplace=True)
# feture engineering a new feature "TotalFS"

housing['TotalSF'] = (housing['TotalBsmtSF'] + housing['1stFlrSF'] + housing['2ndFlrSF'])

housing['Total_sqr_SF'] = (housing['BsmtFinSF1'] + housing['BsmtFinSF2'] + housing['1stFlrSF'] + housing['2ndFlrSF'])

housing['Total_Bathrooms'] = (housing['FullBath'] + (0.5 * housing['HalfBath']) + housing['BsmtFullBath'] + (0.5 * housing['BsmtHalfBath']))

housing['Total_porch_SF'] = (housing['OpenPorchSF'] + housing['3SsnPorch'] + housing['EnclosedPorch'] + housing['ScreenPorch'])
# extract number of years till date from year features

year_features = ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd','YrSold']

for year in year_features:

    housing[year] = int(date.today().year)-housing[year]

datasetShape(housing)
# extract numerical and categorical for dummy and scaling later

numerical_features, categorical_features = divideFeatures(housing)

for feat in categorical_features.columns:

    dummyVars = pd.get_dummies(housing[feat], drop_first=True, prefix=feat+"_")

    housing = pd.concat([housing, dummyVars], axis=1)

    housing.drop(feat, axis=1, inplace=True)

datasetShape(housing)
# shuffle samples

df_shuffle = housing.sample(frac=1, random_state=seed).reset_index(drop=True)
df_y = df_shuffle.pop('SalePrice')

df_X = df_shuffle



# split into train dev and test

X_train, X_test, y_train, y_test = skms.train_test_split(df_X, df_y, train_size=0.7, random_state=seed)

print(f"Train set has {X_train.shape[0]} records out of {len(df_shuffle)} which is {round(X_train.shape[0]/len(df_shuffle)*100)}%")

print(f"Test set has {X_test.shape[0]} records out of {len(df_shuffle)} which is {round(X_test.shape[0]/len(df_shuffle)*100)}%")
from sklearn.feature_selection import RFE

import sklearn.linear_model as sklm
scaler = skp.StandardScaler()



# apply scaling to all numerical variables except dummy variables as they are already between 0 and 1

X_train[numerical_features.columns] = scaler.fit_transform(X_train[numerical_features.columns])



# scale test data with transform()

X_test[numerical_features.columns] = scaler.transform(X_test[numerical_features.columns])



# view sample data

X_train.describe()
# Running RFE to extract top 50 features

lm = sklm.LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 50)

rfe = rfe.fit(X_train, y_train)

rfeCols = X_train.columns[rfe.support_]

X_train_rfe = X_train[rfeCols]

X_test_rfe = X_test[rfeCols]

print("Selected features by RFE are",list(rfeCols))
# plotting mean test and train scoes with alpha 

import operator

def plotCvResults(model_cv):

    cv_results = pd.DataFrame(model_cv.cv_results_)

    cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

    plt.plot(np.log1p(cv_results['param_alpha']), cv_results['mean_train_score'])

    plt.plot(np.log1p(cv_results['param_alpha']), cv_results['mean_test_score'])

    plt.xlabel('log1p(alpha)')

    plt.ylabel('Negative Mean Absolute Error')

    plt.title("Negative Mean Absolute Error and log1p(alpha)")

    plt.legend(['train score', 'test score'], loc='upper left')

    plt.show()



# display parameters

def bestParams(model):

    print("Best Alpha for Regularized Regression:",model.get_params()['alpha'])

    model_parameters = [abs(x) for x in list(model.coef_)]

    model_parameters.insert(0, model.intercept_)

    model_parameters = [round(x, 3) for x in model_parameters]

    cols = X_train_rfe.columns

    cols = cols.insert(0, "constant")

    model_coef = sorted(list(zip(cols, model_parameters)), key=operator.itemgetter(1), reverse=True)[:11]

    print("Top 10 Model parameters (excluding constant) are:")

    for p,c in model_coef:

        print(p)

        

def modelR2AndSpread(model):

    y_train_pred = model.predict(X_train_rfe)

    print("Train r2:",skm.r2_score(y_true=y_train, y_pred=y_train_pred))

    y_test_pred = model.predict(X_test_rfe)

    print("Test r2:",skm.r2_score(y_true=y_test, y_pred=y_test_pred))

    print('Root Mean Square Error train: ' + str(np.sqrt(skm.mean_squared_error(y_train, y_train_pred))))

    print('Root Mean Square Error test: ' + str(np.sqrt(skm.mean_squared_error(y_test, y_test_pred)))) 



    fig = plt.figure(figsize=(16,10))

    plt.suptitle("Linear Regression Assumptions", fontsize = 16)



    # plot error spread

    fig.add_subplot(2, 2, 1)

    sns.regplot(y_train, y_train_pred)

    plt.title('y_train vs y_train_pred spread', fontsize = 14)

    plt.xlabel('y_train', fontsize = 12)

    plt.ylabel('y_train_pred', fontsize = 12)      



    fig.add_subplot(2, 2, 2)

    sns.regplot(y_test, y_test_pred)

    plt.title('y_test vs y_test_pred spread', fontsize = 14)

    plt.xlabel('y_test', fontsize = 12)

    plt.ylabel('y_test_pred', fontsize = 12)      



    # plot residuals for linear regression assumption

    residuals_train = y_train - y_train_pred

    fig = plt.figure(figsize=(16,6))

    fig.add_subplot(2, 2, 3)

    sns.distplot(residuals_train)

    plt.title('residuals between y_train & y_train_pred', fontsize = 14)

    plt.xlabel('residuals', fontsize = 12)



    fig.add_subplot(2, 2, 4)

    residuals_test = y_test - y_test_pred

    sns.distplot(residuals_train)

    plt.title('residuals between y_test & y_test_pred', fontsize = 14)

    plt.xlabel('residuals', fontsize = 12)

    plt.show()
lmr = sklm.Ridge(alpha=0.001)

lmr.fit(X_train_rfe, y_train)



# predict

y_train_pred = lmr.predict(X_train_rfe)

print("Train r2:",skm.r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lmr.predict(X_test_rfe)

print("Test r2:",skm.r2_score(y_true=y_test, y_pred=y_test_pred))
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 1.0, 5.0, 10]}

ridge = sklm.Ridge()



# cross validation

model_cv_ridge = skms.GridSearchCV(estimator = ridge, n_jobs=-1, param_grid = params, 

                             scoring= 'neg_mean_squared_error', cv = 5, 

                             return_train_score=True, verbose = 3)            

model_cv_ridge.fit(X_train_rfe, y_train)

plotCvResults(model_cv_ridge)
# verify log1p value for best selected alpha by GridSearch

print(model_cv_ridge.best_params_['alpha'])

print(np.log1p(model_cv_ridge.best_params_['alpha']))
lml = sklm.Lasso(alpha=0.01)

lml.fit(X_train_rfe, y_train)



# predict

y_train_pred = lml.predict(X_train_rfe)

print("Train r2:",skm.r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lml.predict(X_test_rfe)

print("Test r2:",skm.r2_score(y_true=y_test, y_pred=y_test_pred))
# list of alphas to tune

params = {'alpha': [0.00005, 0.0001, 0.0005, 0.0008, 0.001, 0.005, 0.01, 0.05, 0.1]}

lasso = sklm.Lasso()



# cross validation

model_cv_lasso = skms.GridSearchCV(estimator = lasso, n_jobs=-1, param_grid = params, 

                             scoring= 'neg_mean_squared_error', cv = 5, 

                             return_train_score=True, verbose = 3)            

model_cv_lasso.fit(X_train_rfe, y_train)

plotCvResults(model_cv_lasso)
# verify log1p value for best selected alpha by GridSearch

print(model_cv_lasso.best_params_['alpha'])

print(np.log1p(model_cv_lasso.best_params_['alpha']))
alpha = model_cv_ridge.best_params_['alpha']

ridge_final = sklm.Ridge(alpha=alpha)



ridge_final.fit(X_train_rfe, y_train)

bestParams(ridge_final)
# r2 score for selected model

modelR2AndSpread(ridge_final)
alpha = model_cv_lasso.best_params_['alpha']

lasso_final = sklm.Lasso(alpha=alpha)



lasso_final.fit(X_train_rfe, y_train)

bestParams(lasso_final)
# r2 score for selected model

modelR2AndSpread(lasso_final)