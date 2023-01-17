import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns #plotting package

color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew 
from scipy.special import boxcox1p #for Box Cox transformation
from sklearn.preprocessing import LabelEncoder

#Regressors
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
import lightgbm as lgb

#Pipeline related
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel

#Base classes to be inherited
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

#Cross Validation related
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

#Model Tuning related
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
pd.options.display.max_rows = 80

from subprocess import check_output

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_ID = train['Id']
test_ID = test['Id']

#We don't need ID for predictions
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
test.dtypes
def plot_dist_norm(dist, title):
    sns.distplot(dist, fit=norm);
    (mu, sigma) = norm.fit(dist);
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title(title)
    fig = plt.figure()
    res = stats.probplot(dist, plot=plt)
    plt.show()
plot_dist_norm(train['SalePrice'], 'SalePrice Distribution')
transform_log = np.log1p(train["SalePrice"])
transform_boxcox = boxcox1p(train["SalePrice"], 0.15)

plot_dist_norm(transform_log, 'log(SalePrice) Distribution')
plot_dist_norm(transform_boxcox, 'boxcox(SalePrice) Distribution')
train["SalePrice"] = transform_log
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = np.expm1(train['SalePrice']))
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (np.expm1(train['SalePrice'])<300000)].index)
all_data = pd.concat((train, test)).reset_index(drop=True)
temp = all_data.copy()
temp.drop(['SalePrice'], axis=1, inplace=True)
all_data_na = temp.isnull().sum()
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
all_data_missing = pd.DataFrame({'Missing Numbers' :all_data_na})
all_data_missing
all_data_na =  all_data_na / len(temp)*100
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
all_data['SaleType'].value_counts()
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['SaleType'].value_counts()
all_data['KitchenQual'].value_counts()
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['KitchenQual'].value_counts()
plot_dist_norm(all_data['BsmtFinSF1'].dropna(), 'BsmtFinSF1')
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)
plot_dist_norm(all_data['BsmtFinSF1'], 'BsmtFinSF1')
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)
plot_dist_norm(all_data['BsmtFinSF2'], 'BsmtFinSF2')
plot_dist_norm(all_data['GarageCars'].dropna(), 'GarageCars')
all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
plot_dist_norm(all_data['GarageCars'], 'GarageCars')
plot_dist_norm(all_data['GarageArea'].dropna(), 'GarageArea')
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
plot_dist_norm(all_data['GarageArea'], 'GarageArea')
plot_dist_norm(all_data['TotalBsmtSF'].dropna(), 'TotalBsmtSF')
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
plot_dist_norm(all_data['TotalBsmtSF'], 'TotalBsmtSF')
all_data['Exterior2nd'].value_counts()
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['Exterior2nd'].value_counts()
all_data['Exterior1st'].value_counts()
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior1st'].value_counts()
plot_dist_norm(all_data['BsmtUnfSF'].dropna(), 'BsmtUnfSF')
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)
plot_dist_norm(all_data['BsmtUnfSF'], 'BsmtUnfSF')
all_data['Electrical'].value_counts()
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['Electrical'].value_counts()
all_data['Functional'].value_counts()
all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])
all_data['Functional'].value_counts()
all_data['Utilities'].value_counts()
test['Utilities'].value_counts()
all_data.drop(['Utilities'], axis=1, inplace=True)
plot_dist_norm(all_data['BsmtHalfBath'].dropna(), 'BsmtHalfBath')
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)
plot_dist_norm(all_data['BsmtHalfBath'], 'BsmtHalfBath')
plot_dist_norm(all_data['BsmtFullBath'].dropna(), 'BsmtFullBath')
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)
plot_dist_norm(all_data['BsmtFullBath'], 'BsmtFullBath')
all_data['MSZoning'].value_counts()
all_data['MSZoning']=all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['MSZoning'].value_counts()
plot_dist_norm(all_data['MasVnrArea'].dropna(), 'MasVnrArea')
all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)
plot_dist_norm(all_data['MasVnrArea'], 'MasVnrArea')
all_data['MasVnrType'].value_counts()
all_data['MasVnrType']=all_data['MasVnrType'].fillna('None')
all_data['MasVnrType'].value_counts()
all_data['BsmtFinType1'].value_counts()
all_data['BsmtFinType1']=all_data['BsmtFinType1'].fillna('None')
all_data['BsmtFinType1'].value_counts()
all_data['BsmtFinType2'].value_counts()
all_data['BsmtFinType2']=all_data['BsmtFinType2'].fillna('None')
all_data['BsmtFinType2'].value_counts()
all_data['BsmtQual'].value_counts()
all_data['BsmtQual']=all_data['BsmtQual'].fillna('None')
all_data['BsmtQual'].value_counts()
all_data['BsmtCond'].value_counts()
all_data['BsmtCond']=all_data['BsmtCond'].fillna('None')
all_data['BsmtCond'].value_counts()
all_data['BsmtExposure'].value_counts()
all_data['BsmtExposure']=all_data['BsmtExposure'].fillna('None')
all_data['BsmtExposure'].value_counts()
all_data['GarageType'].value_counts()
all_data['GarageType']=all_data['GarageType'].fillna('None')
all_data['GarageType'].value_counts()
plot_dist_norm(all_data['GarageYrBlt'].dropna(), 'GarageYrBlt')
all_data['GarageYrBlt'][all_data['GarageYrBlt']>2150]
all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(all_data['YearBuilt'][ all_data['GarageYrBlt'].isnull()])
all_data['GarageYrBlt'][all_data['GarageYrBlt']>2018] = all_data['YearBuilt'][all_data['GarageYrBlt']>2018]
plot_dist_norm(all_data['GarageYrBlt'], 'GarageYrBlt')
all_data['GarageFinish'].value_counts()
all_data['GarageFinish']=all_data['GarageFinish'].fillna('None')
all_data['GarageFinish'].value_counts()
all_data['GarageCond'].value_counts()
all_data['GarageCond']=all_data['GarageCond'].fillna('None')
all_data['GarageCond'].value_counts()
all_data['GarageQual'].value_counts()
all_data['GarageQual']=all_data['GarageQual'].fillna('None')
all_data['GarageQual'].value_counts()
plot_dist_norm(all_data['LotFrontage'].dropna(), 'LotFrontage')
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
plot_dist_norm(all_data['LotFrontage'], 'LotFrontage')
all_data['FireplaceQu'].value_counts()
all_data['FireplaceQu']=all_data['FireplaceQu'].fillna('None')
all_data['FireplaceQu'].value_counts()
all_data['Fence'].value_counts()
all_data['Fence']=all_data['Fence'].fillna('None')
all_data['Fence'].value_counts()
all_data['Alley'].value_counts()
all_data['Alley']=all_data['Alley'].fillna('None')
all_data['Alley'].value_counts()
all_data['MiscFeature'].value_counts()
all_data['MiscFeature']=all_data['MiscFeature'].fillna('None')
all_data['MiscFeature'].value_counts()
all_data['PoolQC'].value_counts()
all_data['PoolQC']=all_data['PoolQC'].fillna('None')
all_data['PoolQC'].value_counts()
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].astype(str)
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)
all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)
all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)
all_data['GarageCars'] = all_data['GarageCars'].astype(str)
all_data['TotalSF']=all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']
cols = ('ExterCond','HeatingQC', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold','GarageYrBlt','YearBuilt','YearRemodAdd', 'BsmtHalfBath','BsmtFullBath', 'GarageCars')
    
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
all_data.FireplaceQu = all_data.FireplaceQu.astype('category', ordered=True, categories=['None','Po','Fa','TA','Gd','Ex']).cat.codes
all_data.BsmtQual = all_data.BsmtQual.astype('category', ordered=True, categories=['None','Fa','TA','Gd','Ex']).cat.codes
all_data.BsmtCond = all_data.BsmtCond.astype('category', ordered=True, categories=['None','Po','Fa','TA','Gd']).cat.codes
all_data.GarageQual = all_data.GarageQual.astype('category', ordered=True, categories=['None','Po','Fa','TA','Gd','Ex']).cat.codes
all_data.GarageCond = all_data.GarageCond.astype('category', ordered=True, categories=['None','Po','Fa','TA','Gd','Ex']).cat.codes
all_data.ExterQual = all_data.ExterQual.astype('category', ordered=True, categories=['Fa','TA','Gd','Ex']).cat.codes
all_data.PoolQC = all_data.PoolQC.astype('category', ordered=True, categories=['None','Fa','Gd','Ex']).cat.codes
all_data.KitchenQual = all_data.KitchenQual.astype('category', ordered=True, categories=['Fa','TA','Gd','Ex']).cat.codes
skewed_feats = all_data[all_data.dtypes[all_data.dtypes != "object"].index].apply(lambda x: skew(x.dropna())).drop('SalePrice').sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=skewed_feats.index, y=skewed_feats)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Skewness', fontsize=15)
plt.title('Skewness by feature', fontsize=15)
for feat in skewness[abs(skewness)>0.5].index:
    all_data[feat] = boxcox1p(all_data[feat], 0.15)
all_data = pd.get_dummies(all_data)
new_train = all_data[all_data['SalePrice'].notnull()]
new_test=all_data[all_data['SalePrice'].isnull()]
new_test.drop('SalePrice', axis=1, inplace=True)
corrmat = new_train.corr()
features = corrmat.nlargest(52, 'SalePrice')['SalePrice'].index
sns.set(font_scale=1.2)
plt.subplots(figsize=(12,9))
relevant_features = features[:21]
sns.heatmap(new_train[relevant_features].corr(), cbar=True, annot=True, fmt='.2f', annot_kws={'size':10}, yticklabels=relevant_features.values, xticklabels=relevant_features.values, vmax=1, square=True)
relevant_features = ['SalePrice']
relevant_features = np.append(relevant_features,features[21:41])
sns.set(font_scale=1.2)
plt.subplots(figsize=(12,9))
sns.heatmap(new_train[relevant_features].corr(), cbar=True, annot=True, fmt='.2f', annot_kws={'size':10}, yticklabels=relevant_features, xticklabels=relevant_features, vmax=1, square=True)
relevant_features = ['SalePrice']
relevant_features = np.append(relevant_features,features[41:])
sns.set(font_scale=1.2)
plt.subplots(figsize=(12,9))
sns.heatmap(new_train[relevant_features].corr(), cbar=True, annot=True, fmt='.2f', annot_kws={'size':10}, yticklabels=relevant_features, xticklabels=relevant_features, vmax=1, square=True)
pairing = features[:11]
sns.pairplot(new_train[pairing], size=2.5)
pairing=['SalePrice']
pairing = np.append(pairing,features[11:21])
sns.pairplot(new_train[pairing], size=2.5)
pairing=['SalePrice']
pairing = np.append(pairing,features[21:31])
sns.pairplot(new_train[pairing], size=2.5)
pairing=['SalePrice']
pairing = np.append(pairing,features[31:41])
sns.pairplot(new_train[pairing], size=2.5)
pairing=['SalePrice']
pairing = np.append(pairing,features[41:51])
sns.pairplot(new_train[pairing], size=2.5)
pairing=['SalePrice']
pairing = np.append(pairing,features[51:])
sns.pairplot(new_train[pairing], size=2.5)
y_train = new_train.SalePrice.values
new_train.drop('SalePrice', axis=1, inplace=True)
nfold = 5

def cv_score(model):
    kf = KFold(nfold, shuffle=True, random_state=42).get_n_splits(new_train.values)
    return np.sqrt(-cross_val_score(model, new_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
class AveragedModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.dot(predictions,self.weights)      
    
class StackedAveragedModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, nfold=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.nfold = nfold

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.nfold, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)                
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
flag=True
if flag:    
    steps = [('scaler',RobustScaler()),('select',SelectFromModel(Lasso(alpha =5e-4, random_state=1,selection='cyclic'))),
            ('lasso', Lasso(alpha =5e-4, random_state=1,selection='cyclic'))]
    lasso_p = Pipeline(steps)
    alphas=(np.linspace(1e-4,1e-3,num=10))
    selections=['random','cyclic']
    gscv = GridSearchCV(lasso_p, cv=nfold, param_grid={'lasso__alpha': alphas, 'lasso__selection': selections}, n_jobs=1, verbose=1)
    gscv.fit(new_train.values, y_train)
    lasso_ = gscv.best_estimator_.named_steps.lasso
    score = cv_score(lasso_)
    print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    print('Best Lasso: ',lasso_)
    lasso = make_pipeline(RobustScaler(),SelectFromModel(lasso_), lasso_)

    steps = [('scaler',RobustScaler()),('select',SelectFromModel(ElasticNet(alpha=1e-3, 
            l1_ratio=5e-1, random_state=3, selection='random'))),('enet', ElasticNet(alpha=1e-3, 
            l1_ratio=5e-1, random_state=3, selection='random'))]
    enet_p = Pipeline(steps)
    alphas=np.linspace(1e-3,1e-2,num=10)
    l1_ratios=np.linspace(0,1,num=11)
    selections=['random','cyclic']
    gscv = GridSearchCV(enet_p, cv=nfold,
                      param_grid={'enet__alpha': alphas, 'enet__l1_ratio': l1_ratios, 'enet__selection':selections }, n_jobs=1, verbose=1)
    gscv.fit(new_train.values, y_train)
    enet_ = gscv.best_estimator_.named_steps.enet
    score = cv_score(enet_)
    print("\nElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    print('Best ElasticNet: ',enet_)
    enet = make_pipeline(RobustScaler(),SelectFromModel(enet_), enet_)


    steps = [('scaler',RobustScaler()),('select',SelectFromModel(lasso_)),('krr',KernelRidge(alpha=6e-1, 
        kernel='polynomial', degree=2, coef0=4))]
    krr_p = Pipeline(steps)
    alphas=np.linspace(1e-1,1,num=10)
    degrees=[1,2,3]
    coef0s = [1,2,4,8]

    gscv = GridSearchCV(krr_p, cv=nfold,
                      param_grid={'krr__alpha': alphas, 'krr__degree': degrees, 'krr__coef0':coef0s }, n_jobs=1, verbose=1)
    gscv.fit(new_train.values, y_train)
    krr_ = gscv.best_estimator_.named_steps.krr 
    score = cv_score(krr_)
    print("\n Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    print('Best Kernel Ridge: ',krr_)
    krr = make_pipeline(RobustScaler(),SelectFromModel(lasso_), krr_)

    steps = [('scaler',RobustScaler()),('select',SelectFromModel(lasso_)),('gboost',GradientBoostingRegressor(n_estimators=2000,
            learning_rate=2e-2, max_depth=5, max_features='log2', 
            min_samples_leaf=5, min_samples_split=15, loss='huber', random_state =5))]

    gboost_p = Pipeline(steps)
    n_estimatorss=[1000,2000,3000]
    learning_rates=np.linspace(1e-2,1e-1,num=10)
    max_depths=[1,2,3,4,5]
    max_featuress=['sqrt','log2']
    min_samples_leafs=[5,10,15,20]
    min_samples_splits=[5,10,15,20]
    rscv = RandomizedSearchCV(gboost_p, cv=nfold,
            param_distributions={'gboost__n_estimators': n_estimatorss, 'gboost__learning_rate': learning_rates, 'gboost__max_depth': max_depths, 
            'gboost__min_samples_leaf': min_samples_leafs, 'gboost__min_samples_split': min_samples_splits, 'gboost__max_features': max_featuress},
            n_jobs=1, verbose=1, random_state=7)
    rscv.fit(new_train.values, y_train)
    gboost_ = rscv.best_estimator_.named_steps.gboost
    score = cv_score(gboost_)
    print("\n Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    print('Best Gradient Boosting: ',gboost_)
    gboost = make_pipeline(RobustScaler(),SelectFromModel(lasso_), gboost_)
    
    steps = [('scaler',RobustScaler()),('select',SelectFromModel(lasso_)),('xgboost', xgb.XGBRegressor(colsample_bytree=6e-1, gamma=1e-2,
            learning_rate=4e-2, max_depth=3, min_child_weight=1.5, n_estimators=1000, reg_alpha=4e-1, reg_lambda=9e-1, subsample=4e-1, 
            silent=1, random_state =7,n_jobs = 1))]

    xgboost_p = Pipeline(steps)
    n_estimatorss=[1000,2000,3000]
    learning_rates=np.linspace(1e-2,1e-1,num=10)
    max_depths=[1,2,3,4,5]
    gammas=np.linspace(1e-2,1e-1,num=10)
    min_child_weights=[1.5,1.6,1.7,1.8,1.9]
    colsample_bytrees=np.linspace(1e-1,1,num=10)
    reg_alphas=[0.3,0.4,0.5]
    reg_lambdas=[0.7,0.8,0.9]
    subsamples=[0.3,0.4,0.5,0.6,0.7]
    rscv = RandomizedSearchCV(xgboost_p, cv=nfold,
            param_distributions={'xgboost__n_estimators': n_estimatorss, 'xgboost__learning_rate': learning_rates, 
            'xgboost__max_depth': max_depths, 'xgboost__gamma': gammas, 'xgboost__min_child_weight': min_child_weights, 
            'xgboost__colsample_bytree': colsample_bytrees, 'xgboost__reg_alpha' : reg_alphas, 'xgboost__reg_lambda': reg_lambdas,
            'xgboost__subsample': subsamples},
            n_jobs=1, verbose=1, random_state=7)
    rscv.fit(new_train.values, y_train)
    xgboost_ = rscv.best_estimator_.named_steps.xgboost
    score = cv_score(xgboost_)
    print("\n X Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    print('Best X Gradient Boosting: ',xgboost_)
    xgboost = make_pipeline(RobustScaler(),SelectFromModel(lasso_), xgboost_)
    
    steps = [('scaler',RobustScaler()),('select',SelectFromModel(lasso_)),('lgbm',
            lgb.LGBMRegressor(objective='regression',num_leaves=5, learning_rate=6e-2, n_estimators=500, max_bin = 60, 
            bagging_fraction = 1, bagging_freq = 5, feature_fraction = 1e-1, feature_fraction_seed=9, bagging_seed=9,
            min_data_in_leaf =6, min_sum_hessian_in_leaf = 10, verbose=-1))]

    lgbm_p = Pipeline(steps)
    n_estimatorss=[500,600,700,800,900,1000]
    learning_rates=np.linspace(1e-2,1e-1,num=10)
    num_leavess=[5,10,15]
    max_bins=[10,20,30,40,50,60,70,80,90,100]
    bagging_fractions=[0.2,0.4,0.6,0.8,1]
    bagging_freqs=[1,3,5,7,9]
    feature_fractions=np.linspace(1e-1,1,num=10)
    min_data_in_leafs=[2,4,6,8,10]
    min_sum_hessian_in_leafs=[10,20,30]
    rscv = RandomizedSearchCV(lgbm_p, cv=nfold,
            param_distributions={'lgbm__n_estimators': n_estimatorss, 'lgbm__learning_rate': learning_rates, 
            'lgbm__num_leaves': num_leavess, 'lgbm__max_bin': max_bins, 'lgbm__bagging_fraction': bagging_fractions, 
            'lgbm__bagging_freq': bagging_freqs, 'lgbm__feature_fraction' : feature_fractions, 'lgbm__min_data_in_leaf': min_data_in_leafs,
            'lgbm__min_sum_hessian_in_leaf': min_sum_hessian_in_leafs},
            n_jobs=1, verbose=1, random_state=7)
    rscv.fit(new_train.values, y_train)
    lgbm_ = rscv.best_estimator_.named_steps.lgbm
    score = cv_score(lgbm_)
    print("\n Light Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    print('Best Light Gradient Boosting: ',lgbm_)
    lgbm = make_pipeline(RobustScaler(),SelectFromModel(lasso_), lgbm_)
if flag:
    score = cv_score(lasso)
    lasso_w = 1.0/score.std()
    print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = cv_score(enet)
    enet_w = 1.0/score.std()
    print("\nElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = cv_score(krr)
    krr_w = 1.0/score.std()
    print("\n Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = cv_score(gboost)
    gboost_w = 1.0/score.std()
    print("\n Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = cv_score(xgboost)
    xgboost_w = 1.0/score.std()
    print("\n X Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = cv_score(lgbm)
    lgbm_w = 1.0/score.std()
    print("\n Light Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    weights = [enet_w, gboost_w, krr_w, lasso_w, xgboost_w, lgbm_w]

    averaged_model = AveragedModel(models = (enet, gboost, krr, lasso, xgboost, lgbm), weights=weights/sum(weights))
    score = cv_score(averaged_model)
    averaged_model_w = 1.0/score.std()
    print("\nAveraged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    stacked_averaged_model = StackedAveragedModel(base_models = (lasso, enet, gboost, krr, xgboost, lgbm), meta_model = lasso)
    score=cv_score(stacked_averaged_model)
    stacked_averaged_model_w = 1.0/score.std()
    print("\nStacked Averaged model score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

if flag:
    stacked_averaged_model.fit(new_train.values, y_train)
    stacked_train_pred = stacked_averaged_model.predict(new_train.values)
    stacked_pred = stacked_averaged_model.predict(new_test.values)
    print("\nStacked Averaged model train prediction error: {:.4f} \n".format(np.sqrt(mean_squared_error(y_train, stacked_train_pred))))

    averaged_model.fit(new_train.values, y_train)
    averaged_train_pred = averaged_model.predict(new_train.values)
    averaged_pred = averaged_model.predict(new_test.values)
    print("\nAveraged base model train prediction error: {:.4f} \n".format(np.sqrt(mean_squared_error(y_train, averaged_train_pred))))

    total_weight = stacked_averaged_model_w+averaged_model_w

    train_pred = (stacked_train_pred*stacked_averaged_model_w + averaged_train_pred*averaged_model_w)/total_weight

    print("\nStacked and Averaged train prediction error: {:4f} \n".format(np.sqrt(mean_squared_error(y_train, train_pred))))

    #prediction = (stacked_pred*stacked_averaged_model_w + averaged_pred*averaged_model_w)/total_weight
    prediction = stacked_pred

    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = np.expm1(prediction)
    sub.to_csv('submission.csv',index=False)