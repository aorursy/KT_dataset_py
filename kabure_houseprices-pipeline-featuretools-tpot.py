import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import re
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
# Concatenating the dataframes in only one
df_train['set'] = 'train'
df_test['set'] = 'test'
df_test["SalePrice"] = np.nan
data = pd.concat([df_train, df_test], sort=True)
#Looking  data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
for c in ['MiscFeature', 'Alley', 'Fence']:
    data[c].fillna('none', inplace=True)
    
numerical_feats = data.dtypes[data.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = data.dtypes[data.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))
print(data.shape)

n = data.select_dtypes(include=object)
for column in n.columns:
    print(column, ':  ', data[column].unique())
## Let's see the distribuition of the categories: 
for category in list(categorical_feats):
    print('#'*35)    
    print('Distribuition of feature:', category)
    print(data[category].value_counts(normalize=True))
    print('#'*35)
fig, axes = plt.subplots(ncols=4, nrows=4, 
                         figsize=(4 * 4, 4 * 4), sharey=True)

axes = np.ravel(axes)

cols = ['OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual',
        'BsmtCond','GarageQual','GarageCond', 'MSSubClass','MSZoning',
        'Neighborhood','BldgType','HouseStyle','Heating','Electrical','SaleType']

for i, c in zip(np.arange(len(axes)), cols):
    ax = sns.boxplot(x=c, y='SalePrice', data=data, ax=axes[i])
    ax.set_title(c)
    ax.set_xlabel("")
# to categorical feature
cols = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr",
        "KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd",
        "LowQualFinSF","GarageYrBlt"]

for c in cols:
    data[c] = data[c].astype(str)

# encode quality
# Ex(Excellent), Gd（Good）, TA（Typical/Average）, Fa（Fair）, Po（Poor）
cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
for c in cols:
    data[c].fillna(0, inplace=True)
    data[c].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
def pair_features_to_dummies(df, col1, col2, prefix):
    d_1 = pd.get_dummies(df[col1].astype(str), prefix=prefix)
    d_2 = pd.get_dummies(df[col2].astype(str), prefix=prefix)
    for c in list(set(list(d_1.columns) + list(d_2.columns))):
        if not c in d_1.columns: d_1[c] = 0
        if not c in d_2.columns: d_2[c] = 0
    return (d_1 + d_2).clip(0, 1)

cond = pair_features_to_dummies(data,'Condition1','Condition2','Condition')
exterior = pair_features_to_dummies(data,'Exterior1st','Exterior2nd','Exterior')
bsmtftype = pair_features_to_dummies(data,'BsmtFinType1','BsmtFinType2','BsmtFinType') 

all_data = pd.concat([data, cond, exterior, bsmtftype], axis=1)
all_data.drop(['Condition1','Condition2', 'Exterior1st','Exterior2nd','BsmtFinType1','BsmtFinType2'], axis=1, inplace=True)
all_data.head()

# fillna
for c in ['MiscFeature', 'Alley', 'Fence']:
    data[c].fillna('None', inplace=True)
    
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

data.loc[data.GarageYrBlt.isnull(),'GarageYrBlt'] = data.loc[all_data.GarageYrBlt.isnull(),'YearBuilt']

data['GarageType'].fillna('None', inplace=True)
data['GarageFinish'].fillna(0, inplace=True)

for c in ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    data[c].fillna(0, inplace=True)
from sklearn.preprocessing import LabelEncoder


for i, t in data.loc[:, data.columns != 'SalePrice'].dtypes.iteritems():
    if t == object:
        data[i].fillna(data[i].mode()[0], inplace=True)
        data[i] = LabelEncoder().fit_transform(data[i].astype(str))
    else:
        data[i].fillna(data[i].median(), inplace=True)
data['OverallQualCond'] = data['OverallQual'] * data['OverallCond']
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['Interaction'] = data['TotalSF'] * data['OverallQual']
df_train = data[data['SalePrice'].notnull()]
df_test = data[data['SalePrice'].isnull()].drop('SalePrice', axis=1)
df_test.head()
fig, axes = plt.subplots(ncols=4, nrows=9, figsize=(20, 30))
axes = np.ravel(axes)
col_name = df_train.corr()['SalePrice'][1:].index
for i in range(36):
    df_train.plot.scatter(ax=axes[i], x=col_name[i], 
                          y='SalePrice', c='OverallQual', 
                          sharey=True, colorbar=False, cmap='viridis')
df_train = df_train[df_train['TotalSF'] < 6000]
df_train = df_train[df_train['TotalBsmtSF'] < 4000]
df_train = df_train[df_train['SalePrice'] < 700000]
X_train = df_train.drop(['SalePrice','Id'], axis=1).values
y_train = df_train['SalePrice'].values
X_test  = df_test.drop(['Id'], axis=1).values

print(X_train.shape, y_train.shape, X_test.shape)
########################################################
######## IMPORTING NECESSARY MODULES AND MODELS ########
########################################################
from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.metrics import explained_variance_score, median_absolute_error, r2_score, mean_squared_error #To evaluate our model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split # Model evaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler # Preprocessing
from sklearn.linear_model import Lasso, Ridge, ElasticNet, RANSACRegressor, SGDRegressor, HuberRegressor, BayesianRidge # Linear models
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor  # Ensemble methods
from xgboost import XGBRegressor, plot_importance # XGBoost
from sklearn.svm import SVR, SVC, LinearSVC  # Support Vector Regression
from sklearn.tree import DecisionTreeRegressor # Decision Tree Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline # Streaming pipelines
from sklearn.decomposition import KernelPCA, PCA # Dimensionality reduction
from sklearn.feature_selection import SelectFromModel # Dimensionality reduction
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV # Model evaluation
from sklearn.base import clone # Clone estimator
from sklearn.metrics import mean_squared_error as MSE
thresh = 5 * 10**(-4)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
#select features using threshold
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)
# eval model
select_X_val = selection.transform(X_test)
# test 
select_X_test = selection.transform(X_test)
select_X_train.shape
pipelines = []
seed = 5

pipelines.append(
                ("Scaled_Ridge", 
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Ridge", Ridge(random_state=seed, alpha= 0.1, tol=0.1, solver='auto' ))]
                 )))

pipelines.append(
                ("Scaled_Lasso", 
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Lasso", Lasso(random_state=seed, tol=0.1))]
                 )))

pipelines.append(
                ("Scaled_Elastic", 
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Lasso", ElasticNet(random_state=seed, tol=0.1))]
                 )))

pipelines.append(
                ("Scaled_RF_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("RF", RandomForestRegressor(random_state=seed))]
                 )))

pipelines.append(
                ("Scaled_ET_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("ET", ExtraTreesRegressor(random_state=seed))]
                 )))

pipelines.append(
                ("Scaled_BR_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("BR", BaggingRegressor(random_state=seed))]
                 ))) 

pipelines.append(
                ("Scaled_Hub-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Hub-Reg", HuberRegressor())]
                 ))) 

pipelines.append(
                ("Scaled_BayRidge",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("BR", BayesianRidge())]
                 ))) 

pipelines.append(
                ("Scaled_XGB_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("XGBR", XGBRegressor(seed=seed, n_estimators=300))]
                 ))) 

pipelines.append(
                ("Scaled_DT_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("DT_reg", DecisionTreeRegressor())]
                 ))) 

pipelines.append(
                ("Scaled_SVR",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("SVR",  SVR(kernel='linear', C=1e3, degree=2))]
                 )))

pipelines.append(
                ("Scaled_KNN_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("KNN_reg", KNeighborsRegressor())]
                 )))
pipelines.append(
                ("Scaled_ADA-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("ADA-reg", AdaBoostRegressor())
                 ]))) 

pipelines.append(
                ("Scaled_Gboost-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("GBoost-Reg", GradientBoostingRegressor())]
                 )))

pipelines.append(
                ("Scaled_RFR_PCA",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=3)),
                     ("XGB", RandomForestRegressor())]
                 )))

pipelines.append(
                ("Scaled_XGBR_PCA",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=3)),
                     ("XGB", XGBRegressor())]
                 )))

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'r2'
n_folds = 7

results, names  = [], [] 

for name, model  in pipelines:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, select_X_train, y_train, cv= kfold,
                                 scoring=scoring, n_jobs=1)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn Name", fontsize=20)
ax.set_ylabel("R Squared Score of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()
import featuretools as ft #importing the module
from featuretools import variable_types as vtypes # importing vtypes to classify or categoricals
es = ft.EntitySet("house_price") #Creating new Entityset
#Seting the categorical and ordinal variables
house_variable_types = {
    'BldgType': vtypes.Categorical, 'BsmtCond': vtypes.Categorical, 'BsmtExposure': vtypes.Categorical, 
    'BsmtFinType1': vtypes.Categorical, 'BsmtFinType2': vtypes.Categorical, 'BsmtQual': vtypes.Ordinal, 
    'CentralAir': vtypes.Categorical, 'Id':vtypes.Categorical, 'Exterior2nd': vtypes.Categorical, 
    'Condition1': vtypes.Categorical, 'Condition2': vtypes.Categorical, 'Electrical': vtypes.Categorical,
    'ExterCond': vtypes.Categorical, 'ExterQual': vtypes.Ordinal, 'Exterior1st': vtypes.Categorical, 
    'Foundation': vtypes.Categorical, 'Functional': vtypes.Categorical, 'GarageCond': vtypes.Categorical, 
    'GarageFinish': vtypes.Categorical, 'GarageQual': vtypes.Ordinal, 'GarageType': vtypes.Categorical, 
    'Heating': vtypes.Categorical, 'HeatingQC': vtypes.Categorical, 'HouseStyle': vtypes.Categorical, 
    'LandContour': vtypes.Categorical, 'LandSlope': vtypes.Categorical, 'LotConfig': vtypes.Categorical, 
    'LotShape': vtypes.Categorical, 'MSZoning': vtypes.Categorical, 'MasVnrType': vtypes.Categorical, 
    'Neighborhood': vtypes.Categorical, 'PavedDrive': vtypes.Categorical,'RoofMatl': vtypes.Categorical,
    'RoofStyle': vtypes.Categorical, 'SaleCondition': vtypes.Categorical, 'SaleType': vtypes.Categorical, 
    'Street': vtypes.Categorical, 'MiscFeature': vtypes.Categorical, 'KitchenQual': vtypes.Ordinal, 
    'Utilities': vtypes.Categorical, 'OverallQual': vtypes.Ordinal, 'PoolQC': vtypes.Categorical, 
    'Alley': vtypes.Categorical, 'FireplaceQu': vtypes.Categorical
}
#Creating a new entity from our table (data) with Id and we will put the correct variable types
es.entity_from_dataframe(entity_id="NewFeatures",
                         dataframe=data, index="Id",
                         variable_types=house_variable_types)
print(data.shape)
# Creating a new entity using the OverallQuality and the most correlated with our target variables
es.normalize_entity('NewFeatures', 'Quality', 'OverallQual',
                    additional_variables=['Neighborhood','GarageQual','SaleCondition',
                                          'KitchenQual','HouseStyle', 'Condition1'],
                    make_time_index=False)
### Need I set the PriceSale in any moment? 
# es.add_interesting_values(max_values=3)
feature_matrix, features = ft.dfs(entityset=es, 
                                  target_entity="NewFeatures", 
                                  max_depth=2, verbose=True)
feature_matrix.shape
#Let's drop some of outliers 
feature_matrix = feature_matrix[feature_matrix['TotalSF'] < 6000]
feature_matrix = feature_matrix[feature_matrix['TotalBsmtSF'] < 4000]
feature_matrix.shape
feature_matrix = feature_matrix.reset_index() ## I am reseting to try fix  error
feature_matrix = feature_matrix.fillna(-999) ## filling NA's 

df_train = feature_matrix[feature_matrix['set'] == 1].copy() # spliting the data into df train
df_train = df_train[feature_matrix['SalePrice'] < 700000] # EXcluding some outliers 

df_test = feature_matrix[feature_matrix['set'] == 0].copy() # spliting the data into df test

#Deleting some inutil features (SalePrice in df_test was just to better handle with the full dataset)
del df_test['SalePrice']
del df_train['set']
del df_test['set']
## Why I got back NaN and/or inifite values? 
X_train = df_train.drop(['SalePrice','Id'], axis=1).values
y_train = df_train['SalePrice'].values
X_test  = df_test.drop(['Id'], axis=1).values

print(X_train.shape, y_train.shape, X_test.shape)
thresh = 5 * 10**(-4)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
#select features using threshold
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)
# eval model
select_X_val = selection.transform(X_test)
# test 
select_X_test = selection.transform(X_test)
print(select_X_train.shape)
pipelines = []
seed = 5

pipelines.append(
                ("Scaled_Ridge", 
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Ridge", Ridge(random_state=seed, tol=1 ))]
                 )))

pipelines.append(
                ("Scaled_Lasso", 
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Lasso", Lasso(random_state=seed, tol=0.1))]
                 )))

pipelines.append(
                ("Scaled_Elastic", 
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Lasso", ElasticNet(random_state=seed, tol=0.1))]
                 )))

pipelines.append(
                ("Scaled_SVR",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("SVR",  SVR(kernel='linear', C=1e2, degree=5))]
                 )))

pipelines.append(
                ("Scaled_RF_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("RF", RandomForestRegressor(random_state=seed))]
                 )))

pipelines.append(
                ("Scaled_ET_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("ET", ExtraTreesRegressor(random_state=seed))]
                 )))

pipelines.append(
                ("Scaled_BR_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("BR", BaggingRegressor(random_state=seed))]
                 ))) 

pipelines.append(
                ("Scaled_Hub-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Hub-Reg", HuberRegressor())]
                 ))) 

pipelines.append(
                ("Scaled_BayRidge",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("BR", BayesianRidge())]
                 ))) 

pipelines.append(
                ("Scaled_XGB_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("XGBR", XGBRegressor(seed=seed))]
                 ))) 

pipelines.append(
                ("Scaled_DT_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("DT_reg", DecisionTreeRegressor())]
                 ))) 

pipelines.append(
                ("Scaled_KNN_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("KNN_reg", KNeighborsRegressor())]
                 )))
pipelines.append(
                ("Scaled_ADA-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("ADA-reg", AdaBoostRegressor())
                 ]))) 

pipelines.append(
                ("Scaled_Gboost-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("GBoost-Reg", GradientBoostingRegressor())]
                 )))

pipelines.append(
                ("Scaled_RFR_PCA",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=3)),
                     ("XGB", RandomForestRegressor())]
                 )))

pipelines.append(
                ("Scaled_XGBR_PCA",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=3)),
                     ("XGB", XGBRegressor())]
                 )))

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'r2'
n_folds = 7

results, names  = [], [] 

for name, model  in pipelines:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, select_X_train, y_train, cv= kfold,
                                 scoring=scoring)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn Name", fontsize=20)
ax.set_ylabel("R Squared Score of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()
# Importing the necessary library
from tpot import TPOTRegressor
## It's a implementation of some customized models to do in future
tpot_config = {
    'sklearn.ensemble.GradientBoostingRegressor': {
        ''
    },
    'xgboost.XGBRegressor': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    }
}

# We will create our TPOT regressor with commonly used arguments
tpot = TPOTRegressor(verbosity=2, scoring='r2', cv=3, 
                      n_jobs=-1, generations=6, config_dict='TPOT light',
                      population_size=50, random_state=3,
                      early_stop = 5)
# Fitting the auto ML model
#fiting our tpot auto model
tpot.fit(select_X_train, y_train)

