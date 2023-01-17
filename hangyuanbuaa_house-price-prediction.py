# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.decomposition import PCA

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
print(train_data.shape)

print(test_data.shape)
n_train = train_data.shape[0]

n_test = test_data.shape[0]

housing = pd.concat([train_data.drop('SalePrice',axis=1),test_data]).reset_index(drop=True)

#housing = pd.read_csv('../input/train.csv')
print(housing.shape)

housing.tail()
#housing['SalePrice'].hist(bins=50)

sns.distplot(train_data['SalePrice'], fit=norm);

(mu, sigma) = norm.fit(train_data['SalePrice'])

plt.legend(['Norm Distribution: $\mu$={:.2f}, $\sigma$={:.2f}'.format(mu, sigma)], loc='best')

plt.ylabel('Frequncy')

plt.figure()

res = stats.probplot(train_data['SalePrice'], plot=plt)

plt.show()
print(train_data['SalePrice'].skew())

print(train_data['SalePrice'].kurt())
train_data['LogSalePrice'] = np.log(train_data['SalePrice'])

sns.distplot(train_data['LogSalePrice'], fit=norm)

(mu, sigma) = norm.fit(train_data['LogSalePrice'])

plt.legend(['Norm Distribution: $\mu$={:.2f}, $\sigma$={:.2f}'.format(mu, sigma)], loc='best')

plt.ylabel('Frequncy')

plt.figure()

stats.probplot(train_data['LogSalePrice'], plot=plt)

plt.show()
data_stats = housing.notnull().sum()

to_drop_cols = data_stats[data_stats<housing.shape[0]*.1].index.tolist()

to_drop_cols # drop features with samples less than 10% of dataset size
housing_clean = housing.copy()

#housing_clean.drop(to_drop_cols, axis=1, inplace=True)

housing_clean.drop(['Id'], axis=1, inplace=True)

housing_clean.shape
num_cols = housing_clean._get_numeric_data().columns.tolist()

#numeric = [feat for feat in num_cols if feat not in ordinal]

housing[num_cols].describe()
housing[num_cols].hist(figsize=(20,18), bins=30);
housing_clean[['YearBuilt', 'YearRemodAdd']].isnull().any()
housing_clean['HouseAge'] = 2019 - housing_clean['YearBuilt']

housing_clean['RemodSince'] = 2019 - housing_clean['YearRemodAdd']

print(housing_clean[['YearBuilt','YearRemodAdd','HouseAge','RemodSince']].sample(5))

housing_clean.drop(['YearBuilt','YearRemodAdd'], axis=1, inplace=True)
housing_clean[['MSSubClass']] = housing_clean[['MSSubClass']].astype('category')
missing_data_stats = housing_clean.isnull().sum()/housing_clean.shape[0] * 100

missing_data_stats[missing_data_stats>0.1].sort_values(ascending=False).plot(kind='bar');
cols = missing_data_stats.sort_values(ascending=False).nlargest(5).index

train_data[cols].describe()
housing_clean.loc[housing['BsmtQual'].isnull().values,['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']].sample(10)
bsmt_cat_cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']

bsmt_num_cols = ['BsmtFinSF1','BsmtUnfSF','TotalBsmtSF']

housing_clean[bsmt_cat_cols].isnull().sum()
housing_clean.loc[housing_clean['BsmtFinType1'].isnull().values,bsmt_cat_cols].isnull().all()
housing_clean.loc[housing_clean['BsmtFinType1'].isnull().values, bsmt_cat_cols] = 'None'

housing_clean.loc[housing_clean['BsmtFinType1'].isnull().values, bsmt_num_cols] = 0
garage_cat_cols = ['GarageType','GarageFinish','GarageQual','GarageCond']

garage_num_cols = ['GarageYrBlt','GarageCars','GarageArea']

housing_clean[garage_cat_cols].isnull().sum()
housing_clean.loc[housing_clean['GarageType'].isnull().values,garage_cat_cols].isnull().all()
housing_clean.loc[housing_clean['GarageType'].isnull().values, garage_cat_cols] = 'None'

housing_clean.loc[housing_clean['GarageType'].isnull().values, garage_num_cols] = 0
#housing_clean['PoolQC'] = housing_clean['PoolQC'].fillna('None')

#housing_clean['MiscFeature'] = housing_clean['MiscFeature'].fillna('None')

#housing_clean['Alley'] = housing_clean['Alley'].fillna('None')

#housing_clean['Fence'] = housing_clean['Fence'].fillna('None')

#housing_clean['FireplaceQu'] = housing_clean['FireplaceQu'].fillna('None')
missing_data_stats = housing_clean.isnull().sum()

cols = missing_data_stats[missing_data_stats>0].index.tolist()
missing_data_stats[missing_data_stats>0]
#housing_clean[housing_clean['FireplaceQu'].isnull()]
cat_cols = housing_clean.select_dtypes(exclude=['int64', 'float64']).columns

housing_clean[cat_cols].describe()
for c in cols:

    if c in cat_cols:

        mode = housing_clean[c].mode()[0]

        housing_clean[c] = housing_clean[c].fillna(mode)

    else:

        median = housing_clean[c].median()

        housing_clean[c] = housing_clean[c].fillna(median)
housing_clean.info()
# key optional features (with NA)

bsmt = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtUnfSF','TotalBsmtSF']

garg = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageCars']

# other optional features

frpl = ['Fireplaces', 'FireplaceQu']

pool = ['PoolArea', 'PoolQC']

fenc = ['Fence'] # quality

misc = ['MiscFeature', 'MiscVal'] 

porch = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']

# common features 

bath = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']

bdrm = ['Bedroom']

ktcn = ['Kitchen','KitchenQual']
#housing_clean[housing_clean['PoolArea']>0].PoolArea

#(housing_clean['Fireplaces']>0).sum()

#housing_clean['MiscFeature'].value_counts()

#housing_clean['Fence'].value_counts()
housing_clean['withBasement'] = housing_clean['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)

housing_clean['withGarage'] = housing_clean['GarageType'].apply(lambda x: 1 if x!='None' else 0)

housing_clean['withFireplaces'] = housing_clean['Fireplaces'].apply(lambda x: 1 if x>0 else 0)

housing_clean['withFence'] = housing_clean['Fence'].apply(lambda x: 1 if x!='None' else 0)

housing_clean['withPool'] = housing_clean['PoolQC'].apply(lambda x: 1 if x!='None' else 0)

housing_clean['withMiscFeature'] = housing_clean['MiscFeature'].apply(lambda x: 1 if x!='None' else 0)
housing_clean['numBsmtbath'] = housing_clean['BsmtFullBath'] + 0.5*housing_clean['BsmtHalfBath']

housing_clean['numBath'] = housing_clean['FullBath'] + 0.5*housing_clean['HalfBath']

housing_clean.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'], axis=1, inplace=True)
housing_clean['TotBsmtFinSF'] = housing_clean['BsmtFinSF1']+housing_clean['BsmtFinSF2']

housing_clean['BsmtFinSFRatio'] = housing_clean['TotBsmtFinSF']/housing_clean['TotalBsmtSF']

housing_clean['BsmtUnfSFRatio'] = housing_clean['BsmtUnfSF']/housing_clean['TotalBsmtSF']

housing_clean['BsmtFinSFRatio'] = housing_clean['BsmtFinSFRatio'].fillna(0)

housing_clean['BsmtUnfSFRatio'] = housing_clean['BsmtUnfSFRatio'].fillna(0)

#housing_clean.drop(['BsmtUnfSF','TotBsmtFinSF'], axis=1, inplace=True)

housing_clean.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotBsmtFinSF'], axis=1, inplace=True)
housing_clean['PorchArea'] = housing_clean['OpenPorchSF'] + housing_clean['EnclosedPorch'] 

+ housing_clean['3SsnPorch'] + housing_clean['ScreenPorch']

housing_clean.drop(['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], axis=1, inplace=True)
housing_clean['numFloors'] = housing_clean['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0) + 1

housing_clean['TotalSF'] = housing_clean['1stFlrSF'] + housing_clean['2ndFlrSF'] + housing_clean['TotalBsmtSF'] + housing_clean['GarageArea']

housing_clean['1stFlrSFRatio'] = housing_clean['1stFlrSF']/housing_clean['TotalSF']

housing_clean['2ndFlrSFRatio'] = housing_clean['2ndFlrSF']/housing_clean['TotalSF']

housing_clean.drop(['1stFlrSF','2ndFlrSF'], axis=1, inplace=True)
housing_clean[['Condition1','Condition2','Exterior1st','Exterior2nd']].isnull().any()
dummy1 = pd.get_dummies(housing_clean.Condition1, prefix='Condition')

dummy2 = pd.get_dummies(housing_clean.Condition2, prefix='Condition')

conditions = (dummy1 + dummy2).replace(2, 1)

conditions['Condition_RRNe'] = dummy1['Condition_RRNe']

conditions.head()
dummy1 = pd.get_dummies(housing_clean.Exterior1st, prefix='Exterior')

dummy2 = pd.get_dummies(housing_clean.Exterior2nd, prefix='Exterior')



exteriors = (dummy1 + dummy2).replace(2, 1)

common = dummy1.merge(dummy2, how='outer')

for col in exteriors.columns:

    if exteriors[col].isnull().all():

        exteriors[col] = common[col]
housing_clean.drop(['Condition1','Condition2','Exterior1st','Exterior2nd'], inplace=True, axis=1)

# drop these features first, add the encoded dummy variables back later
cat_cols = housing_clean.select_dtypes(exclude=['int64', 'float64']).columns

cat_levels = housing_clean[cat_cols].nunique()

ordinal = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'OverallCond', 'OverallQual', 'Fence',

           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu',

           'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']



#`Alley` and `Utilities` are actually multi-level categoricals.

binary_cats = list(set(cat_levels[cat_levels==2].index) - set(['Alley','Utilities']))

multi_level_cats = list(set(cat_levels[cat_levels>2].index) - set(ordinal+['OverallQual','OverallCond']))+['Alley','Utilities']
housing_clean[cat_cols] = housing_clean[cat_cols].astype('category')
mapper = {'Street': {'Grvl':0, 'Pave':1}, 'CentralAir': {'N':0, 'Y':1}}

for col in binary_cats:

    housing_clean[col] = housing_clean[col].replace(mapper[col])

housing_clean[binary_cats].head()
# One Hot Encode all categoricals

housing_clean = pd.get_dummies(housing_clean)

housing_clean.shape
housing_clean.head()
level_order = {'LotShape': ['Reg','IR1','IR2','IR3'],

 'LandSlope': ['Gtl','Mod','Sev'],

 'ExterQual': ['Ex','Gd','TA','Fa','Po'],

 'ExterCond': ['Ex','Gd','TA','Fa','Po'],

 'BsmtQual': ['Ex','Gd','TA','Fa','Po','None'],

 'BsmtCond': ['Ex','Gd','TA','Fa','Po','None'],

 'BsmtExposure': ['Gd','Av','Mn','No','None'],

 'BsmtFinType1': ['GLQ','ALQ','BLQ','Rec','LwQ','Unf','None'],

 'BsmtFinType2': ['GLQ','ALQ','BLQ','Rec','LwQ','Unf','None'],

 'HeatingQC': ['Ex','Gd','TA','Fa','Po'],

 'KitchenQual': ['Ex','Gd','TA','Fa','Po'],

 'FireplaceQu': ['Ex','Gd','TA','Fa','Po','None'],

 'GarageFinish': ['Fin','RFn','Unf','None'],

 'GarageQual': ['Ex','Gd','TA','Fa','Po','None'],

 'GarageCond': ['Ex','Gd','TA','Fa','Po','None'],

 'PavedDrive': ['Y','P','N'],

 'PoolQC': ['Ex','Gd','TA','Fa','Po','None'],

 'Fence': ['GdPrv','MnPrv','GdWo','MnWw','None']}

#for col in ordinal:

#    if col not in ['OverallQual', 'OverallCond']:

#        ordered_cat = pd.api.types.CategoricalDtype(ordered=True, categories=level_order[col][::-1])

#        housing_clean[col] = housing_clean[col].astype(ordered_cat)



#for col in ordinal:

#    housing_clean[col].cat.remove_unused_categories()

#    housing_clean[col] = housing_clean[col].cat.codes



#label_mapping = {}

#for c in ordinal:

#    housing[c], label_mapping[c] = pd.factorize(housing[c])
#dummies = pd.get_dummies(housing_clean[multi_level_cats], drop_first=False)

#housing_clean.drop(multi_level_cats, axis=1, inplace=True)

#dummies.head()
housing_clean = housing_clean.join([conditions, exteriors])

housing_clean.shape
to_drop_cols = []

for col in list(housing_clean.columns):

    if 'None' in col:

        to_drop_cols.append(col)

print(to_drop_cols)
housing_clean.drop(to_drop_cols, axis=1, inplace=True)
def split(df):

    X_train = df[:n_train].values

    X_test = df[-n_test:].values

    y_train = train_data.loc[:,'LogSalePrice'].values

    print(X_train.shape, X_test.shape)

    return X_train, y_train, X_test
sc = StandardScaler()

X_train, y_train, X_test = split(housing_clean)

X = sc.fit_transform(X_train)
rf_reg = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)

rf_reg.fit(X, y_train)

scores = cross_val_score(rf_reg, X, y_train, scoring='neg_mean_squared_error', cv=5)

rmse_scores = np.sqrt(-scores)

print(rmse_scores)

#feat_imp = pd.Series(rf_reg.feature_importances_, index=housing_clean.columns)

#feat_imp[feat_imp>0.002].sort_values(ascending=False).plot(kind='bar');
#housing_feats = housing_clean[feat_imp[feat_imp>0.002].index.tolist()]

#X_train, y_train, X_test = split(housing_feats)

#X = sc.fit_transform(X_train)
def pca_plot(pca, X):

    cumsum = np.cumsum(pca.explained_variance_ratio_)

    d = np.argmax(cumsum>=0.95)+1

    n_components = X.shape[1]

    idx = np.arange(1, n_components+1)

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(idx, cumsum)

    ax.axvline(x=d, linestyle=':', color='r')

    ax.axhline(y=0.95, linestyle=':', color='r')

    ax.set_xlabel('# Principal Components')

    ax.set_ylabel('Explained Variance')

    ax.set_xlim([-1, n_components])

    plt.show()

    print('95% Variance explained by {} PCs.'.format(d))
pca = PCA()

pca.fit(X)

pca_plot(pca, X)
dim = ['PC{}'.format(i) for i in range(1, len(pca.components_)+1)]

components = pd.DataFrame(pca.components_, columns=housing_clean.keys())

components.index=dim

components.head()
ratios = pca.explained_variance_ratio_.reshape(len(pca.components_),1)

variance_ratios = pd.DataFrame(ratios, columns=['explained_variance'])

variance_ratios.index = dim

variance_ratios.shape
pca.components_.shape, len(pca.components_), len(dim)
def plot_PCs(pc, k, ax):

    base_color = sns.color_palette()[0]

    features = pc.reindex(pc.abs().sort_values(ascending=False).index)

    features[:k].plot(ax=ax, kind='bar', color=base_color)
fig, ax = plt.subplots(figsize=(10,4))

plot_PCs(components.loc['PC1',:], 20, ax)
pca = PCA(n_components=0.95)

X_reduced = pca.fit_transform(X)
X_reduced.shape
plt.scatter(X_reduced[:,0], y_train);
rf_reg = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)

rf_reg.fit(X_reduced, y_train)
#Select the most important PCs:

#pc_imp = pd.Series(rf_reg.feature_importances_, index=list(range(1,len(pca.components_)+1)))

#pc_imp[pc_imp>0.0015].sort_values(ascending=False).plot(kind='bar');

#X_selected = X_reduced[:,pc_imp[pc_imp>0.0015].sort_values(ascending=False).index.tolist()]

#X.shape, X_reduced.shape, X_selected.shape
def rmse(reg, X, y):

    y_pred = reg.predict(X)

    mse = mean_squared_error(y, y_pred)

    rmse = np.sqrt(mse)

    return rmse
#rf_reg = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=100)

rf_reg.fit(X, y_train)

print('Score with all features: {:.4f}, RMSE = {:.5f}'.format(rf_reg.score(X, y_train), rmse(rf_reg, X, y_train)))

rf_reg.fit(X_reduced, y_train)

print('Score with 141 PCs: {:.4f}, RMSE = {:.5f}'.format(rf_reg.score(X_reduced, y_train), rmse(rf_reg, X_reduced, y_train)))

#rf_reg.fit(X_selected, y_train)

#print('Score with selected PCs: {:.4f}, RMSE = {:.5f}'.format(rf_reg.score(X_selected,y_train), rmse(rf_reg, X_selected, y_train)))
# Cross Validation

scores = cross_val_score(rf_reg, X_reduced, y_train, scoring='neg_mean_squared_error', cv=5)

rmse_scores = np.sqrt(-scores)

print(rmse_scores)
def CV_results(search):

    res = search.cv_results_

    for mean_score, params in zip(res['mean_test_score'], res['params']):

        print('RMSE is {:.4f} w/: {}'.format(np.sqrt(-mean_score),params))

    print(search.best_estimator_)
param_grid = [

    {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30], 'min_samples_leaf': [5, 10], 'random_state':[0]},

    {'bootstrap': [False], 'n_estimators': [10, 50], 'max_depth': [6, 8, 10, 20], 'min_samples_leaf': [2, 5, 10]}

]

rf_reg = RandomForestRegressor()

grid_search = GridSearchCV(rf_reg, param_grid[0], cv=5, scoring='neg_mean_squared_error')
# trained with PCs

grid_search.fit(X_reduced, y_train)

CV_results(grid_search)
best_model_1 = grid_search.best_estimator_
# trained with all features in training dataset

#grid_search.fit(X, y_train)

#CV_results(grid_search)
#best_model_2 = grid_search.best_estimator_
from sklearn.linear_model import Lasso

lin_reg = Lasso(alpha=0.0034)

lin_reg.fit(X_reduced, y_train)
scores = cross_val_score(lin_reg, X_reduced, y_train, scoring='neg_mean_squared_error', cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores)
param_grid = [

    {'alpha': np.linspace(0.0001, 0.01, 10)}

]

lasso_reg = Lasso()

grid_search = GridSearchCV(lasso_reg, param_grid[0], cv=5, scoring='neg_mean_squared_error')
#grid_search.fit(X_reduced, y_train)

#CV_results(grid_search)
from sklearn.svm import LinearSVR, SVR



svm_reg = LinearSVR(epsilon=1.5)

svm_reg.fit(X, y_train)
svm_poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)

svm_poly_reg.fit(X, y_train)
svm_k_reg = SVR(kernel='rbf', C=100, gamma=0.01)

svm_k_reg.fit(X, y_train)
scores = cross_val_score(svm_k_reg, X_reduced, y_train, scoring='neg_mean_squared_error', cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores)
param_grid = [

    {'C': np.linspace(0.1, 1, 3), 'gamma': np.linspace(0.0001, 0.0006, 3)}

]

svm_reg = SVR(kernel='rbf')

grid_search = GridSearchCV(svm_reg, param_grid[0], cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_reduced, y_train)

CV_results(grid_search)
best_model_2 = grid_search.best_estimator_
#best_model = grid_search.best_estimator_

X_test_std = sc.transform(X_test)

X_test_reduced = pca.transform(X_test_std)

#y_test_pred = best_model.predict(X_test_reduced)

y_test_pred_1 = best_model_1.predict(X_test_reduced)

y_test_pred_2 = best_model_2.predict(X_test_reduced)
np.exp(y_test_pred_2)
sub = pd.DataFrame()

sub['Id'] = test_data['Id']

sub['SalePrice'] = np.exp(y_test_pred_2)

sub.to_csv('submission.csv', index=False)