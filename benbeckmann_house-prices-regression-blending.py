import sys

import re



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_columns', None)

sns.set_style('ticks', {'axes.grid': True, 'axes.spines.right': False, 'axes.spines.top': False})

sns.set_palette(sns.color_palette("RdBu", n_colors=21))

warnings.filterwarnings('ignore')



from scipy.special import boxcox1p, inv_boxcox1p

from scipy.stats import kruskal, skew, boxcox_normmax



from sklearn.preprocessing import LabelEncoder, RobustScaler, PolynomialFeatures

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import Lasso, Ridge, ElasticNet

from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor

from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from vecstack import stacking
# utility class for updating both train and test sets

class Dataset:

    target = 'sale_price'

    

    def __init__(self):

        _train = pd.read_csv('../input/train.csv')

        _train.columns = _train.columns.map(self.format_column_name)

        

        _test = pd.read_csv('../input/test.csv')

        _test.columns = _test.columns.map(self.format_column_name)

        self.test_id = _test.id

        

        _train.drop('id', axis=1, inplace=True)

        _test.drop('id', axis=1, inplace=True)

        

        self.update(_train, _test)



    def __getitem__(self, key):

        return self.dataset[key]

    

    def __setitem__(self, key, args):

        if self.target in list(key):

            raise

        

        if isinstance(args, tuple):

            self.train[key] = self.train[key].pipe(*args)      

            self.test[key] = self.test[key].pipe(*args)

            self.update(self.train, self.test)

        else:

            self.dataset[key] = args

            self.update()

    

    def __len__(self):

        return len(self.dataset)

    

    def format_column_name(self, s):

        s = re.findall('[A-Z, 0-9][a-z, 0-9]*', s)

        s = '_'.join(s).lower()

        return s

    

    def update(self, *args):

        if len(args):

            self.train, self.test = args

            self.dataset = pd.concat([self.train, self.test], axis=0)

        else:

            self.train = self.dataset[:len(self.train)]

            self.test = self.dataset[len(self.train):]

        

        for key in self.dataset.columns:

            self.__dict__[key] = self.dataset[key]



        global train, test

        train = self.train

        test = self.test





dataset = Dataset()
pd.DataFrame([train.columns, train.dtypes, train.isnull().sum(), train.nunique()])
pd.DataFrame([test.columns, test.dtypes, test.isnull().sum(), test.nunique()])
# manual check for features

ordinals = ['exter_qual', 'exter_cond', 'bsmt_qual', 'bsmt_cond', 'heating_q_c',

            'kitchen_qual', 'fireplace_qu', 'garage_qual', 'garage_cond', 'pool_q_c', 

            'overall_qual', 'overall_cond', 'm_s_sub_class']

years = ['year_built', 'year_remod_add', 'garage_yr_blt', 'yr_sold']

nominals = list(set(train.select_dtypes('object').columns) - set(ordinals) - set(years)) + ['mo_sold']

continuous = list(set(train.select_dtypes(['int64', 'float64']).columns) - set(ordinals) - set(years) - set(nominals))

continuous.remove('sale_price')



len(ordinals) + len(years) + len(nominals) + len(continuous) + 1 == len(train.columns)
# normalize sale_price

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

sns.distplot(train.sale_price, ax=ax[0])



boxcox_lambda = boxcox_normmax(train['sale_price'] + 1)

train['sale_price'] = boxcox1p(train['sale_price'], boxcox_lambda)

sns.distplot(train.sale_price, ax=ax[1])



dataset.update(train, test)
data = dataset[ordinals].isnull().sum() / len(dataset)

data = data[data != 0].sort_values()

plt.figure(figsize=(10, 5))

sns.barplot(data.values, data.index, orient='h') 

# fill NA appropriately
for feat in ordinals:

    print(f'{feat}: {sorted(train[feat].dropna().unique())}')
dataset['kitchen_qual'] = pd.Series.fillna, train.kitchen_qual.mode()[0]

dataset[data.index] = pd.DataFrame.fillna, 'NA'



# map/encode ordinals

mapping = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

dataset[ordinals[:-3]] = pd.DataFrame.applymap, lambda x: mapping.get(x)

dataset[ordinals[-3:]] = pd.DataFrame.apply, LabelEncoder().fit_transform
fig, ax = plt.subplots(3, 5, figsize=(15, 10))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in ordinals:

    data = train[[feat, 'sale_price']]

    sns.boxplot(x=feat, y='sale_price', data=data, ax=next(ax))

# the higher the quality, the higher the price
plt.figure(figsize=(15, 5))

sns.heatmap(train[ordinals + ['sale_price']].corr(), annot=True)
data = dataset[years].isnull().sum() / len(dataset)

data = data[data != 0].sort_values()

print(data)

# fill NA appropriately
# fill with min decade

dataset['garage_yr_blt'] = pd.Series.fillna, 0
fig, ax = plt.subplots(2, 2, figsize=(30, 20))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in years:

    sns.boxplot(x='sale_price', y=feat, orient='h', data=train, ax=next(ax))
fig, ax = plt.subplots(2, 2, figsize=(15, 5))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in years:

    sns.lineplot(x=feat, y='sale_price', data=train[train[feat] != 0], ax=next(ax))
# try visualizing by decade

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in years[:-1]:

#     data = train.groupby(train[feat] // 10 * 10).agg({'sale_price': 'mean'}).reset_index()

    data = train.copy(deep=True)

    data[feat]= data[feat] // 10 * 10

    sns.lineplot(x=feat, y='sale_price', data=data[data[feat] != 0], ax=next(ax))
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in years[:-1]:

    data = train[[feat, 'sale_price']]

    data[feat] = data[feat] // 10 * 10

    sns.boxplot(x='sale_price', y=feat, orient='h', data=data, ax=next(ax))
# bin years to decade

dataset[years[:-1]] = pd.DataFrame.apply, lambda x: x // 10 * 10
data = dataset[continuous].isnull().sum() / len(dataset)

data = data[data != 0].sort_values()
data = dataset[continuous].isnull().sum() / len(dataset)

data = data[data != 0].sort_values()

plt.figure(figsize=(10, 5))

sns.barplot(data.values, data.index, orient='h') 

# fill NA appropriately
dataset['lot_frontage'] = pd.Series.fillna, train.lot_frontage.median()

dataset[data.index] = pd.DataFrame.fillna, 0
fig, ax = plt.subplots(6, 5, figsize=(15, 10))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in sorted(continuous):

    sns.scatterplot(x=feat, y='sale_price', data=train, ax=next(ax))
plt.figure(figsize=(25, 10))

sns.heatmap(train[continuous + ['sale_price']].corr(), annot=True)
data = dataset[nominals].isnull().sum() / len(dataset)

data = data[data != 0].sort_values()

plt.figure(figsize=(10, 5))

sns.barplot(data.values, data.index, orient='h')

# fill NA appropriately
# fill train and test with train mode

mode_fills = ['m_s_zoning', 'functional', 'utilities', 'exterior1st', 'electrical', 'exterior2nd', 'sale_type']

dataset[mode_fills] = pd.DataFrame.fillna, train[mode_fills].mode().iloc[0]



# fill the rest with NA

dataset[data.index] = pd.DataFrame.fillna, 'NA'
fig, ax = plt.subplots(7, 5, figsize=(15, 25))

fig.tight_layout()

ax = iter(ax.flatten())

for feat in nominals:

    data = train[[feat, 'sale_price']]

    # sort by median for better visualization

    labels = data.groupby(feat)['sale_price'].median().sort_values().index

    sns.boxplot(x='sale_price', y=feat, orient='h', data=data, order=labels, ax=next(ax))
dataset['total_sf'] = dataset.gr_liv_area + dataset.total_bsmt_s_f

dataset['bsmt_comp'] = dataset.bsmt_qual * dataset.bsmt_cond * dataset.total_bsmt_s_f

dataset['garage_comp'] = dataset.garage_cars * dataset.garage_qual * dataset.garage_cond

dataset['total_bath'] = dataset.full_bath + dataset.bsmt_full_bath + dataset.half_bath + dataset.bsmt_half_bath

dataset['total_porch_sf'] = dataset.open_porch_s_f + dataset['3_ssn_porch'] + dataset.enclosed_porch + dataset.screen_porch + dataset.wood_deck_s_f

continuous.extend(['total_sf', 'bsmt_comp', 'garage_comp', 'total_bath', 'total_porch_sf'])
def normalize(df, feats):

    skewed = df[feats].apply(skew)

    normalized = list()

    for feat, skewness in skewed.items():

        if skewness > 0.5:

            n = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1)).tolist()

            normalized.append(n)

        else:

            normalized.append(df[feat].tolist())

    return np.array(normalized).T



def get_outlier_indices(df, feats):

    indices = list()

    for feat in feats:

        col = df[feat]

        Q1 = np.percentile(col, 25)

        Q3 = np.percentile(col, 75)

        IQR = Q3 - Q1

        step = IQR * 1.5

        idx = col[(col < Q1 - step) | (col > Q3 + step)].index

        indices.extend(idx)

    return list(set(indices))



def get_kruskals(df, feats):

    # non-parametric one way anova to measure association 

    # between nominal and continuous variables

    stats = list()

    target = 'sale_price'

    for feat in feats:

        groups = df.groupby(feat).agg({target: list})[target].tolist()

        stat = kruskal(*groups)

        stats.append([feat, stat.statistic, stat.pvalue])

    stats = pd.DataFrame(stats, columns=['feat', 'stat', 'pvalue'])

    return stats.sort_values('stat', ascending=False)
# normalize

normalize_feats = continuous + ordinals

dataset[normalize_feats] = normalize(dataset, normalize_feats)



# drop outliers

outlier_feats = ['total_sf']

train = train[~train.index.isin(get_outlier_indices(train, outlier_feats))]

dataset.update(train, test)
continuous = train[continuous + ['sale_price']].corr()['sale_price'].sort_values(ascending=False)

continuous = continuous[continuous > 0.3].index.tolist()[1:]

print(continuous)
ordinals = train[ordinals + ['sale_price']].corr()['sale_price'].sort_values(ascending=False)

ordinals = ordinals[ordinals > 0.3].index.tolist()[1:]

print(ordinals)
stats = get_kruskals(train, nominals)

print(stats[:10])



nominals = stats.feat.tolist()[:10]



dataset['has_bsmt'] = dataset['total_bsmt_s_f'] > 0

dataset['has_garage'] = dataset['garage_area'] > 0

dataset['has_fireplace'] = dataset['fireplaces'] > 0

nominals.extend(['has_bsmt', 'has_garage', 'has_fireplace'])
# try to check if there's a trend over mo-yr sold, otherwise exclude feature

data =  train[['yr_sold', 'mo_sold', 'sale_price', 'overall_qual']]

data['yr_mo_sold'] = pd.to_datetime(train.yr_sold.astype(str) + '-' + train.mo_sold.astype(str)).dt.to_period('M')

data = data.sort_values('yr_mo_sold')

plt.figure(figsize=(25, 10))

sns.boxplot(x='sale_price', y='yr_mo_sold', orient='h', data=data)



years = years[:-1]
# check other feats

sns.boxplot(x='lot_config', y='sale_price', hue='lot_shape', data=train)

nominals.extend(['lot_config', 'lot_shape'])
# label encode

dataset[years] = pd.DataFrame.apply, LabelEncoder().fit_transform

dataset[nominals] = pd.DataFrame.apply, LabelEncoder().fit_transform



# onehot encode

years_encoded = pd.get_dummies(dataset[years].astype(object), drop_first=True)

nominals_encoded = pd.get_dummies(dataset[nominals].astype(object), drop_first=True)
dataset_preprocessed = pd.concat([

    dataset[continuous], dataset[ordinals],

    nominals_encoded, years_encoded

], axis=1)
dataset_preprocessed.shape
X_train = dataset_preprocessed[:len(train)]

y_train = train.sale_price



X_test = dataset_preprocessed[len(train):]
kfold = KFold(10, shuffle=True)
rmse_scorer = make_scorer(lambda y, y0: np.sqrt(mean_squared_error(y, y0)), greater_is_better=False)
regressors = [

    Ridge(random_state=0),

    Pipeline([('poly', PolynomialFeatures()), ('lasso', Lasso(random_state=0))]),

    Pipeline([('poly', PolynomialFeatures()), ('el', ElasticNet(random_state=0))]),

    RandomForestRegressor(random_state=0),

    ExtraTreesRegressor(random_state=0),

    AdaBoostRegressor(random_state=0),

    GradientBoostingRegressor(random_state=0),

    XGBRegressor(random_state=0)

]



results = list()

for reg in regressors:

    if isinstance(reg, Pipeline):

        name = reg.steps[1][1]

    else:

        name = reg

    name = type(name).__name__

    print(f'Validating {name}..')

    cv = cross_val_score(reg, X_train, y_train, scoring=rmse_scorer, n_jobs=-1, cv=kfold)

    results.append([name, cv])



results = pd.DataFrame(results, columns=['model', 'cv'])

results = results.set_index('model')['cv'].apply(pd.Series)

results['mean'] = results.mean(axis=1)

results['std'] = results.std(axis=1)

results
data = results.unstack().drop(['mean', 'std']).reset_index().rename(columns={0: 'score'})

fig, ax = plt.subplots(2, 1, figsize=(10, 5))

fig.tight_layout()

ax = iter(ax.flatten())

sns.lineplot(x='level_0', y='score', hue='model', data=data, ax=next(ax))

sns.boxplot(x='score', y='model', orient='h', data=data, ax=next(ax))
ridge_params = {

    'alpha': [30, 20, 10, 5, 1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-8, 1e-10, 1e-15]

}



lasso_params = {

    'lasso__alpha': [30, 20, 10, 5, 1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-8, 1e-10, 1e-15]

}



el_params = {

    'el__alpha': [30, 20, 10, 5, 1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-8, 1e-10, 1e-15]

}



rf_params = {

    'max_depth': [5, 10, 15],

    'min_samples_split': [2, 5, 10],

    'n_estimators': [50, 100, 200, 250, 300]

}



et_params = {

    'max_depth': [5, 10, 15],

    'min_samples_split': [2, 5, 10],

    'n_estimators': [50, 100, 200, 250, 300]

}



gb_params = {

    'learning_rate': [0.01, 0.03, 0.05, 0.09, 0.1, 0.2],

    'max_depth': [3, 4, 5, 6],

    'min_samples_split': [2, 5, 10],

    'n_estimators': [100, 200, 300, 400, 500]

}



xgb_params = {

    'learning_rate': [0.01, 0.03, 0.05, 0.09, 0.1, 0.2],

    'max_depth': [3, 4, 5, 6],

    'min_child_weight': [1, 3, 5, 7],

    'gamma': [0.0, 0.1, 0.2 , 0.3, 0.4],

    'colsample_bytree': [0.4, 0.5 , 0.7],

    'n_estimators': [100, 200, 300, 400, 500]

}



params = [ridge_params, lasso_params, el_params, rf_params, et_params, gb_params, xgb_params]
results = list()

for reg, param in zip(regressors[:-3], params[:-2]):

    if isinstance(reg, Pipeline):

        name = reg.steps[1][1]

    else:

        name = reg

    name = type(name).__name__

    print(f'Grid search {name}..')

    reg = GridSearchCV(reg, param, rmse_scorer, n_jobs=-1, cv=kfold, verbose=3)

    reg.fit(X_train, y_train)

    results.append(reg)



for reg, param in zip(regressors[-2:], params[-2:]):

    name = type(reg).__name__

    print(f'Grid search {name}..')

    reg = RandomizedSearchCV(reg, param_distributions=param, scoring=rmse_scorer, n_iter=100, n_jobs=-1, cv=kfold, verbose=3)

    reg.fit(X_train, y_train)

    results.append(reg)
preds = list()

for cv in results:

    reg = cv.best_estimator_

    reg.fit(X_train, y_train)

    pred = inv_boxcox1p(reg.predict(X_train), boxcox_lambda)

    preds.append(pred)

    

fig, ax = plt.subplots(4, 2, figsize=(15, 15))

fig.tight_layout()

ax = iter(ax.flatten())

for pred in preds:

    sns.regplot(x=inv_boxcox1p(y_train, boxcox_lambda), y=pred, ax=next(ax))
fis = list()

for cv in results[3:]:

    fi = cv.best_estimator_.feature_importances_

    fis.append(fi)



fis = pd.DataFrame(fis, columns=X_train.columns)

fis.index = [type(r.best_estimator_).__name__ for r in results[3:]]

fis = fis.unstack().reset_index()

fis = fis.sort_values(['level_1', 0], ascending=[False, False]).groupby('level_1', sort=False).head(5)



plt.figure(figsize=(10, 5))

sns.barplot(x=0, y='level_0', hue='level_1', orient='h', data=fis)
names = ['ridge', 'lasso', 'el', 'rf', 'et', 'gb', 'xgb']

models = dict(zip(names, results))
S_train_1, S_test_1 = stacking(

    [models['rf'].best_estimator_, models['xgb'].best_estimator_, models['el'].best_estimator_], 

    X_train, y_train, X_test, regression=True, n_folds=10, shuffle=True, random_state=0, verbose=2

)



S_train_2, S_test_2 = stacking(

    [models['gb'].best_estimator_, models['et'].best_estimator_, models['ridge'].best_estimator_, models['lasso'].best_estimator_],

    X_train, y_train, X_test, regression=True, n_folds=10, shuffle=True, random_state=0, verbose=2

)
stack_reg_1 = GridSearchCV(Ridge(random_state=0), ridge_params, rmse_scorer, n_jobs=-1, cv=kfold, verbose=3)

stack_reg_1.fit(S_train_1, y_train)



stack_reg_1 = stack_reg_1.best_estimator_.fit(S_train_1, y_train)

stack_1_pred = stack_reg_1.predict(S_test_1)
stack_reg_2 = RandomizedSearchCV(XGBRegressor(random_state=0), param_distributions=xgb_params, scoring=rmse_scorer, n_iter=100, n_jobs=-1, cv=kfold, verbose=3)

stack_reg_2.fit(S_train_2, y_train)



stack_reg_2 = stack_reg_2.best_estimator_.fit(S_train_2, y_train)

stack_2_pred = stack_reg_2.predict(S_test_2)
blended_pred = (

    (0.6 * inv_boxcox1p(stack_1_pred, boxcox_lambda)) +

    (0.4 * inv_boxcox1p(stack_2_pred, boxcox_lambda))

)
# np.random.seed(42)

# weights =x np.random.randint(0, 1000, (50, 4)) # generate 100 weights

# weights = [

#     [round(w, 2) for w in weight / weight.sum()]

#     for weight in weights

# ]

# params = {'weights': weights}

# voting_reg = GridSearchCV(VotingRegressor(estimators), params, rmse_scorer, n_jobs=-1, verbose=3)

# voting_reg.fit(X_train, y_train)



# print(voting_reg.best_score_)

# print(voting_reg.best_params_)



# voting_reg = voting_reg.best_estimator_

# voting_reg.fit(X_train, y_train)

# y_pred = inv_boxcox1p(voting_reg.predict(X_test), boxcox_lambda)
submit = pd.DataFrame({

    'Id': dataset.test_id,

    'SalePrice': blended_pred

})
from IPython.display import FileLink

submit.to_csv('house_prices_submit.csv', index=False)

FileLink('house_prices_submit.csv')