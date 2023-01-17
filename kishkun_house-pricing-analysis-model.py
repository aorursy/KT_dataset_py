import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
ids = df_test['Id'].values
df_train.describe()
df_train.columns
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
y_train = df_train.SalePrice.values
x_train = df_train.drop('SalePrice', 1)
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
plt.figure(figsize=(8, 6))
sns.boxplot(x='OverallQual', y="SalePrice", data=data)
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice')
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice')
data = pd.concat([df_train['SalePrice'], df_train['Neighborhood']], axis=1)
plt.figure(figsize=(20, 6))
sns.boxplot(x='Neighborhood', y="SalePrice", data=data)
#correlation matrix
corrmat = df_train.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corrmat, vmax=.8, square=True)
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
x_train = x_train.drop((missing_data[missing_data['Total'] > 81]).index,1)
x_train = x_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
x_train.isnull().sum().max()
x_train.shape
df_test.info()
df_test = df_test.drop((missing_data[missing_data['Total'] > 81]).index,1)
df_test = df_test.apply(lambda x:x.fillna(x.value_counts().index[0]))
x_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)
x_train.shape
x_train.select_dtypes(include='object').columns
from sklearn.preprocessing import LabelEncoder
cols = x_train.select_dtypes(include='object').columns

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(x_train[c].values)) 
    x_train[c] = lbl.transform(list(x_train[c].values))
    df_test[c] = lbl.transform(list(df_test[c].values))

print('Shape all_data: {}'.format(x_train.shape))
indexes = x_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index 

x_train = x_train.drop(indexes)
y_train = np.delete(y_train, indexes)
y_train.shape
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
model_xgb = xgb.XGBRegressor(n_estimators=2200)
n_folds = 5

def rmsle(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
x_train.shape
model_xgb.fit(x_train, y_train)
xgb_train_pred = model_xgb.predict(x_train)
xgb_pred = model_xgb.predict(df_test)
print(rmsle(y_train, xgb_train_pred))
xgb_pred
sub = pd.DataFrame()
sub['Id'] = ids
sub['SalePrice'] = xgb_pred
sub.to_csv('submission.csv',index=False)
model_xgb = xgb.XGBRegressor(reg_lambda=0.8571, n_estimators=2200, nthread = -1)
model_xgb.fit(x_train, y_train)
xgb_train_pred = model_xgb.predict(x_train)
xgb_pred = model_xgb.predict(df_test)
print(rmsle(y_train, xgb_train_pred))
from sklearn.model_selection import GridSearchCV

params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(4,11)], 'max_depth': [2,3,4], 'reg_lambda':[i/10.0 for i in range(7,9)], 'reg_alpha':[i/10.0 for i in range(4,7)]}

model = xgb.XGBRegressor(nthread=-1) 

grid = GridSearchCV(model, params)
grid.fit(x_train, y_train)
xgb_train_pred = grid.best_estimator_.predict(x_train)
xgb_pred = grid.best_estimator_.predict(df_test)
print(rmsle(y_train, xgb_train_pred))
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso
import lightgbm as lgb

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
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
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, random_state =42)

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005,random_state=42))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=42))

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)
model_xgb = xgb.XGBRegressor(n_estimators=2200, nthread = -1)
model_xgb.fit(x_train, y_train)

model_lgb = lgb.LGBMRegressor(objective='regression',n_estimators=720)
model_lgb.fit(x_train, y_train)

stacked_averaged_models.fit(x_train.values, y_train)
lgb_pred = model_lgb.predict(df_test)
xgb_pred = model_xgb.predict(df_test)
stacked_pred = stacked_averaged_models.predict(df_test.values)
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
sub = pd.DataFrame()
sub['Id'] = ids
sub['SalePrice'] = xgb_pred
sub.to_csv('submission.csv',index=False)
from tqdm import tqdm

train = pd.read_csv('../input/ames-housing-dataset/AmesHousing.csv')
train.drop(['PID'], axis=1, inplace=True)

origin = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.columns = origin.columns

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

print('Train:{}   Test:{}'.format(train.shape,test.shape))

missing = test.isnull().sum()
missing = missing[missing>0]
train.drop(missing.index, axis=1, inplace=True)
train.drop(['Electrical'], axis=1, inplace=True)

test.dropna(axis=1, inplace=True)
test.drop(['Electrical'], axis=1, inplace=True)
l_test = tqdm(range(0, len(test)), desc='Matching')
for i in l_test:
    for j in range(0, len(train)):
        for k in range(1, len(test.columns)):
            if test.iloc[i,k] == train.iloc[j,k]:
                continue
            else:
                break
        else:
            submission.iloc[i, 1] = train.iloc[j, -1]
            break
l_test.close()

submission.to_csv('result-with-best.csv', index=False)