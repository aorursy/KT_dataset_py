# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge, LinearRegression, SGDRegressor, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from scipy.special import boxcox1p
from scipy.stats import norm, skew 
from sklearn.svm import SVR
import xgboost as xgb

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
desc = pd.read_csv("../input/data_description.txt", error_bad_lines=False)
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")
data.head()

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count().sort_values(ascending=False))
missing = pd.concat([total,percent], axis=1, keys=['Total', 'Percent'], sort=False)
missing.head(20)
fig, ax = plt.subplots()
ax.scatter(data['GrLivArea'], data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
data.drop(data[(data['GrLivArea'] > 4300) & (data['SalePrice'] < 300000)].index, inplace=True)
plt.scatter(data['TotalBsmtSF'], data['SalePrice'])
# data.drop(data.loc[data['TotalBsmtSF'] > 3000,:], inplace=True)
# g, ax = plt.subplots(figsize=(20,15))
# sns.heatmap(data.corr(), annot=True, fmt='.2f');
# k = 10 
# corrmat = data.corr()
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(data[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
# pair_cols = ['OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF', 'FullBath', 'YearBuilt', 'YearRemodAdd', 'Foundation_PConc', 'ExterQual_Gd']
# sns.pairplot(data[pair_cols], height=3);
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
sns.distplot(tuple(data['SalePrice']));

plt.subplot(2,2,2)
res = stats.probplot(tuple(data['SalePrice']), plot=plt);

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
data['SalePrice'] = np.log1p(data['SalePrice'])
sns.distplot(data['SalePrice']);
plt.subplot(2,2,2)

res = stats.probplot(data['SalePrice'], plot=plt)


# sns.pairplot(data[list(skewed_features)])

# data['GrLivArea'] = np.log1p(data['GrLivArea'])
# plt.figure(figsize=(10,5))
# plt.subplot(2,2,3)
# sns.distplot(data['GrLivArea']);

# plt.subplot(2,2,4)
# res = stats.probplot(data['GrLivArea'], plot=plt)

# data['BuildEra'] = data['YearBuilt'].apply(lambda x: 'modern' if x > 1985 else('post-world-war' if 1945 < x < 1985 else 'old'))
#data['BuildEra'].value_counts()
# plt.figure(figsize=(10,5))
# plt.subplot(2,2,3)
# sns.distplot(data['YearBuilt']);

# plt.subplot(2,2,4)
# res = stats.probplot(data['YearsBuiltTrans'], plot=plt)
def clean (df):
    """
    Cleaning function
    Takes df as input and outputs a clean df with no missing values and dummy variables for categorical features
    """
    
    # Create a new feature for total area before dropping basment SF
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Categorize build era
    #df['BuildEra'] = df['YearBuilt'].apply(lambda x: 'modern' if x > 1985 else('post-world-war' if 1945 < x < 1985 else 'old'))
    
    # Add feature for has basement or not
    df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0 )
    
    # Drop Features that correlates highly witheachoter or offers nothing
    df.drop(['PoolQC', 'GarageArea', 'MiscFeature', 'MiscVal', 'Alley', 'Fence', 'FireplaceQu', 'MiscFeature', 'TotRmsAbvGrd', 'GarageYrBlt', 'TotalBsmtSF'], axis=1, inplace=True)
    
    # LotFrontage imputation by median of houses in the same neighborhood
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    # Columns for replacing missing values with 'None' category
    cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'MasVnrType' ]
    for col in cols:
        df[col] = df[col].fillna('None')
    
    # Fillna
    df.Electrical.fillna('SBrkr', inplace=True)
    df.MasVnrArea.fillna(0.0, inplace=True)
    
    # Converting numerical features that are really categorical  
    cat_cols = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']
    for cats in cat_cols:
        df[cats] = df[cats].apply(str)


    df = pd.get_dummies(df)
    
    
    
    return df
    
n_train = data.shape[0]
comp = pd.concat([data, test], sort=False)
# Create clean datasets
cleancomp = clean(comp)
cleancomp.isnull().sum().sort_values(ascending=False).head(10)
for i in ('BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'):
    cleancomp[i].fillna(0, inplace=True)

trainset = cleancomp[:n_train]
testset = cleancomp[n_train:]
testset.is_copy = False
trainset.head()
# Split to feature and target
feats = trainset.drop(['SalePrice', 'Id'], axis=1)
label = trainset.SalePrice
testset.drop('SalePrice', axis=1, inplace=True)
testset[testset['TotalSF'].isnull()]
testset.loc[testset['Id'] == 2121, 'TotalSF'] = 896.0
testset.drop(['Id'], axis=1, inplace=True)
# Normalize the data
scaler = StandardScaler()
features = scaler.fit_transform(feats)

X_train, X_test, y_train, y_test = train_test_split(feats, label, test_size=0.3)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(features, label, test_size=0.3)
# # # Random Forest model
# rf = RandomForestRegressor()

# param_dist1 = {"max_depth": [8,10,13],
#                "n_estimators" : [10000],
#                "min_samples_split": [5,6,7,8,],
#                "bootstrap": [True],
#                "min_samples_leaf" : [4,5,6]
#                             }
# scorer = make_scorer(r2_score)
# n_iter_search = 10
# randommodel = RandomizedSearchCV(rf, param_distributions=param_dist1,
#                                    n_iter=n_iter_search, cv=2, scoring=scorer, verbose=10, n_jobs=-1)
# randommodel.fit(features, target);
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.005, max_iter=1000, warm_start=True))
ridge = make_pipeline(MinMaxScaler(feature_range=(0,1)), Ridge())
random = RandomForestRegressor(n_estimators=15000, min_samples_leaf=4, min_samples_split=5, criterion='mse', max_depth=8, warm_start=True, n_jobs=-1)
gradient = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
adaboost = AdaBoostRegressor(n_estimators=10000, learning_rate=0.001)
linear = make_pipeline(RobustScaler(), LinearRegression(n_jobs=-1))
svm = make_pipeline(StandardScaler(), SVR(gamma=0.0001, C=1, verbose=True, kernel='linear'))
# from sklearn.preprocessing import StandardScaler

# params = {
#     'eta0' : [0.00001, 0.001],
#     'alpha': 10.0 ** -np.arange(1, 7),
#     'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
#     'penalty': ['l2', 'l1', 'elasticnet'],
#     'learning_rate': ['constant', 'optimal', 'invscaling']}
# sgd = make_pipeline(RobustScaler(), GridSearchCV(SGDRegressor(), param_grid=params, n_jobs=-1))

elastic = make_pipeline(RobustScaler(), GridSearchCV(ElasticNetCV(), param_grid={
                                                                                 'n_alphas' : [100, 1000],
                                                                                 'l1_ratio' : [.1, .5, .7, .9, .95, .99, 1],
                                                                                 'max_iter' : [1000, 5000]}, cv=3, n_jobs=-1))
# data_matrix = xgb.DMatrix(data=X_train, label=y_train)
# xgb_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 1000)
# xgb_reg.fit(X_train, y_train)
# xgb_pred = xgb_reg.predict(X_test)
# r2_score(y_test, xgb_pred)
# lasso.fit(X_train, y_train)
# lasso_pred = lasso.predict(X_test)
# r2_score(y_test, lasso_pred)
# ridge.fit(X_train, y_train)
# ridge_pred = ridge.predict(X_test)
# r2_score(y_test, ridge_pred)
# random.fit(X_train, y_train)
# predicted_rf = random.predict(X_test)
# r2_score(y_test, predicted_rf)
# gradient.fit(X_train, y_train)
# gradient_pred = gradient.predict(X_test)
# r2_score(y_test, gradient_pred)
# adaboost.fit(X_train, y_train)
# ada_pred = adaboost.predict(X_test)
# r2_score(y_test, ada_pred)
# svm.fit(X_train, y_train)

# svm_preds = svm.predict(X_test)
# r2_score(y_test, svm_preds)
# sgd.fit(X_train, y_train)
# sgd_preds = sgd.predict(X_test)
# r2_score(y_test, sgd_preds)
# elastic.fit(X_train, y_train)
# elastic_preds = elastic.predict(X_test)
# r2_score(y_test, elastic_preds)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
# stackedmodel = AveragingModels(models= (lasso, ridge, random, gradient, adaboost))
# stackedmodel.fit(X_train, y_train)
# stacked_pred = stackedmodel.predict(X_test)
# r2_score(y_test, stacked_pred)
# #Validation function
# n_folds = 5

# def rmsle_cv(model):
#     kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
#     rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
#     return(rmse)
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
metastack = StackingAveragedModels(base_models=(random, gradient, ridge, lasso, linear, svm, elastic), meta_model=lasso)
metastack.fit(feats.values, label.values)
# meta_pred = metastack.predict(X_test.values)
# r2_score(y_test, meta_pred)
predicted_meta = metastack.predict(testset.values)
predicted_meta = np.expm1(predicted_meta)

# random.fit(features, target)
# predictions = random.predict(testset)
df_preds = pd.DataFrame({'Id' : test.Id, 'SalePrice': predicted_meta})
df_preds.head()
os.chdir("/kaggle/working/")
df_preds.to_csv('submission.csv', index=False)


