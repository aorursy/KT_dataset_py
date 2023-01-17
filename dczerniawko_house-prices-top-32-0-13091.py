# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from ml_metrics import rmse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import learning_curve
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from functools import partial
from sklearn.metrics import make_scorer
import lightgbm as lgb
import catboost as ctb
np.random.seed(2018)
# display data
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_row', 1500)
# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_test['SalePrice'] = 0
df_train.head(5)
df_train.info()
# delete Id variable, because it is unnecessary in compute
df_train = df_train.drop(['Id'], axis = 1)
df_train.shape
df_train.describe()
df_train.nunique()
df_train.isnull().any().any()
def missing_values(df):
    for column in df.columns:
        null_rows = df[column].isnull()
        if null_rows.any() == True:
            print('%s: %d nulls' % (column, null_rows.sum()))
missing_values(df_train)
df_train.sample(5)
plt.rcParams['figure.figsize']=(30,20)
sns.heatmap(df_train.corr(method='spearman'), annot=True, linewidths=.5, cmap="Blues");
# five most correlation variable with 'SalePrice'
corr = df_train.corr()
corr['SalePrice'].sort_values(ascending = False)[1:6]
corr['SalePrice'].sort_values(ascending = False)[-5:]
# get correlation matrix, where correlation value is greater than 70%
corr_matrix = df_train.corr(method='spearman')
corr_columns = corr_matrix[corr_matrix[corr_matrix > 0.7] < 1.0].any()
corr_matrix = corr_matrix[corr_columns][corr_columns.index[corr_columns]]
plt.rcParams['figure.figsize']=(15,7)
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap="Blues");
print(df_train.SalePrice.skew())
# A skewness value > 0 means, that there is more weight in the left tail of the distribution.
# target variable
plt.rcParams['figure.figsize']=(10,5)
df_train['SalePrice'].hist();
df_train['SalePrice_bc'], _ = stats.boxcox(df_train['SalePrice'])
df_train['SalePrice_bc'].hist();
df_train['SalePrice_log'] = np.log2( df_train['SalePrice'] + 1)
df_train['SalePrice_log'].hist();
print(np.log2(df_train.SalePrice).skew())
# Normally distributed data, the skewness should be about 0
def good_feats(df):
    feats_from_df = set(df.select_dtypes([np.int]).columns.values)
    bad_feats = {'SalePrice', 'SalePrice_bc'}
    return list(feats_from_df - bad_feats)
def make_hist(df):
    feats = good_feats(df)
    for index, feat in enumerate(feats):
        plt.subplot(len(feats)/5+1, 5, index+1);
        plt.title(feat);
        df[feat].hist();
def make_scatter(df):
    feats = good_feats(df)
    for index, feat in enumerate(feats):
        plt.subplot(len(feats)/5+1, 5, index+1)
        sns.regplot(x=feat, y='SalePrice', data=df_train)
def make_bar(df):
    cat_feats = df_train.select_dtypes(exclude = [np.int, np.float]).columns.values
    cat_feats = cat_feats[:-1]
    for index, feat in enumerate(cat_feats):
        plt.subplot(len(cat_feats)/5+1, 5, index+1)
        sns.barplot(x=feat, y='SalePrice', data=df_train, palette="PRGn");
        plt.xticks(rotation=90);
%%time
plt.figure(figsize=(25,25));
plt.subplots_adjust(hspace = 0.35);

make_hist(df_train)
%%time
plt.figure(figsize=(25,25))
plt.subplots_adjust(hspace = 0.3)

make_scatter(df_train)
%%time
plt.figure(figsize=(20,40))
plt.subplots_adjust(hspace = 0.5, wspace = 0.4)

make_bar(df_train)
df_train.Neighborhood.head(10)
plt.rcParams['figure.figsize']=(15,8)
sns.boxplot(x='Neighborhood', y='SalePrice', data=df_train, palette="PRGn");
plt.xticks(rotation=20);
missing_values(df_train)
df_train['LotFrontage'].head(5)
LFbyN = df_train.groupby('Neighborhood')['LotFrontage'].median().to_dict()
df_train['LotFrontage'] = df_train.apply(lambda row: LFbyN[row['Neighborhood']]
                                                    if pd.isnull(row['LotFrontage'])
                                                    else row['LotFrontage'], axis = 1)
df_train['Electrical'] = df_train['Electrical'].fillna('SBrkr')
cats_nan = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure',  'BsmtFinType1', 'BsmtFinType1', 'BsmtFinType2', 'GarageYrBlt']
for cat in cats_nan:
    df_train[cat] = df_train[cat].fillna("None")
missing_values(df_train)
cat_feats = df_train.select_dtypes(exclude = [np.number]).columns.values
cat_feats
def factorize(df, *columns):
    feats = set(df.select_dtypes(exclude = [np.int, np.float]).columns.values)
    for column in feats:
        df[column + '_cat'] = pd.factorize(df[column])[0]
#factorize(df_train)
#df_train = df_train.select_dtypes(include=[np.number]).interpolate().dropna()
#df_train.head(5)
#df_train.sample(5)
def model_train_predict(model, X, y, success_metric=rmse):
    print('split')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print('fit')
    model.fit(X_train, y_train)
    print('pred')
    y_pred = model.predict(X_test)
    return success_metric(y_test, y_pred)
X = df_train[good_feats(df_train)].values
y = df_train['SalePrice']
model_train_predict(LinearRegression(), X, y)
# the most correlaated variable with target
df_train.OverallQual.isnull().any()
df_train.OverallQual.nunique()
df_train.OverallQual.unique()
df_train.OverallQual.value_counts()
df_train.OverallQual.hist();
plt.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice']);
df_train = df_train[df_train['GrLivArea'] < 4000]
df_test = df_test[df_test['GrLivArea'] < 4000]
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])
# GarageArea - correlation with target: 0.623431
df_train.GarageArea.head(5)
plt.scatter(x=df_train['GarageArea'], y=df_train['SalePrice']);
# GarageCars - correlation with target: 0.640409
df_train.GarageCars.head(5)
plt.scatter(x=df_train['GarageCars'], y=df_train['SalePrice']);
df_train[['GarageCars','GarageArea']].sample(10)
df_train = df_train[df_train['GarageArea'] < 1200]
df_test = df_test[df_test['GarageArea'] < 1200]
#df_train = df_train[df_train['GarageCars'] <= 3]
#df_test = df_test[df_test['GarageCars'] <= 3]
df_train.TotalBsmtSF.sample(5)
plt.scatter(x=df_train['TotalBsmtSF'], y=df_train['SalePrice']);
df_train = df_train[df_train['TotalBsmtSF'] < 3000]
df_test = df_test[df_test['TotalBsmtSF'] < 3000]
df_train.FullBath.sample(5)
plt.scatter(x=df_train['FullBath'], y=df_train['SalePrice']);
df_train.YearBuilt.sample(5)
plt.scatter(x=df_train['YearBuilt'], y=df_train['SalePrice']);
# street 
df_train['Street'].value_counts()
factorize(df_train, 'Street')
factorize(df_test, 'Street')
# sale condition
df_train['SaleCondition'].value_counts()
sns.barplot(x='SaleCondition', y='SalePrice', data=df_train, palette="PRGn");
df_train['SaleCondition'] = df_train['SaleCondition'].apply(lambda x: 1 if x == 'Partial' else 0)
sns.barplot(x='SaleCondition', y='SalePrice', data=df_train, palette="PRGn");
df_train['SaleCondition'].value_counts()
df_train['BsmtQual'].value_counts()
sns.barplot(x='BsmtQual', y='SalePrice', data=df_train, palette="PRGn");
df_train['BsmtQual'] = df_train['BsmtQual'].apply(lambda x: 1 if x == 'Ex' else 0)
df_test['BsmtQual'] = df_test['BsmtQual'].apply(lambda x: 1 if x == 'Ex' else 0)
sns.barplot(x='BsmtQual', y='SalePrice', data=df_train, palette="PRGn");
df_train['KitchenQual'].value_counts()
sns.barplot(x='KitchenQual', y='SalePrice', data=df_train, palette="PRGn");
df_train['KitchenQual'] = df_train['KitchenQual'].apply(lambda x: 1 if x == 'Ex' else 0)
df_test['KitchenQual'] = df_test['KitchenQual'].apply(lambda x: 1 if x == 'Ex' else 0)
sns.barplot(x='KitchenQual', y='SalePrice', data=df_train, palette="PRGn");
df_train['CentralAir'].value_counts()
factorize(df_train, 'CentralAir')
factorize(df_test, 'CentralAir')
df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
df_test['TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']
plt.scatter(x=df_train['TotalSF'], y=df_train['SalePrice']);
df_train.TotalSF.hist();
features = ['OverallQual', 'GrLivArea', 'GarageArea', 'FullBath', 'TotalSF', 'CentralAir_cat', 'KitchenQual_cat']
X = df_train[features].values
y = df_train['SalePrice_log']
model_train_predict(LinearRegression(), X, y)
feats = ['OverallQual', 'GrLivArea', 'GarageArea', 'FullBath', 'TotalSF', 'CentralAir_cat', 'KitchenQual_cat']

X = df_train[feats].values
y = df_train['SalePrice_log'].values

model = LinearRegression()

cv = KFold(n_splits=4)

scores = []
for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = rmse(y_test, y_pred)
    scores.append(score)
    
    
print(np.mean(scores), np.std(scores))
def run_cv(model, X, y, folds=4, target_log=False, cv_type=KFold, success_metric=rmse):
    cv = cv_type(n_splits=folds)
    
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if target_log:
            y_train = np.log(y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if target_log: 
            y_pred = np.exp(y_pred)
        y_pred[y_pred < 0] = 0

        score = success_metric(y_test, y_pred)
        scores.append(score)
        
    return np.mean(scores), np.std(scores)
run_cv(model, X, y, folds=3, target_log='SalePrice_log')
def plot_learning_curve(model, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure(figsize=(12,8))
    plt.title(title)
    if ylim is not None:plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    rmse_scorer = make_scorer(rmse)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=rmse_scorer)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
models = [
    LinearRegression(),
    DecisionTreeRegressor(max_depth=10),
    RandomForestRegressor(max_depth=10),
    ExtraTreesRegressor(max_depth=20)
]

for model in models:
    print(str(model) + ": ")
    %time score = model_train_predict(model, X, y)
    print(str(score) + "\n")
    plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.4, 0.0), n_jobs=4)
    plt.show()
model = LinearRegression()
model.fit(X, y)

weights = list(model.coef_)

dict_feats = {label :weight for label, weight in zip(feats, weights) }
feats = pd.DataFrame([dict_feats])
feats.plot(kind='bar', figsize=(13, 8), title="Feature importances");
models = [
    DecisionTreeRegressor(max_depth=10),
    RandomForestRegressor(max_depth=10),
    ExtraTreesRegressor(max_depth=20)
]

for model in models:
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title('Feature importances: ' + str(model).split('(')[0])
    plt.bar(range(X.shape[1]), model.feature_importances_[indices],
           color = '#00bfff', align = 'center')
    plt.xticks(range(X.shape[1]), [ good_feats(df_train)[x] for x in indices])
    plt.xticks(rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()
feats = ['OverallQual', 'GrLivArea', 'GarageArea', 'FullBath', 'TotalSF', 'CentralAir_cat', 'KitchenQual_cat']
X = df_train[feats].values
y = df_train['SalePrice_log'].values

run_cv(xgb.XGBRegressor(), X, y, folds=4, target_log='SalePrice_log')
X = df_train[feats].values
y = df_train['SalePrice_log'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
def objective(space):    
    xgb_params = {
        'max_depth': int(space['max_depth']),
        'colsample_bytree': space['colsample_bytree'],
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'seed': int(space['seed']),
        'min_child_weight': int(space['min_child_weight']),
        'reg_alpha': space['reg_alpha'],
        'reg_lambda': space['reg_lambda'],
        'n_estimators': 150
    }
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    means_score, std_score = run_cv(xgb.XGBRegressor(), X, y, target_log=True)

    print(means_score, std_score)
    
    print("SCORE: {0}".format(score))
    
    return{'loss':score, 'status': STATUS_OK }

space ={
    'max_depth': hp.quniform ('x_max_depth', 3, 8, 1),
    'colsample_bytree': hp.uniform ('x_colsample_bytree', 0.3, 0.7),
    'learning_rate': hp.uniform ('x_learning_rate', 0.1, 0.3), 
    'subsample': hp.uniform ('x_subsample', 0.3, 0.7),
    'seed': hp.quniform ('x_seed', 0, 10000, 50),
    'min_child_weight': hp.quniform ('x_min_child_weight', 1, 10, 1),
    'reg_alpha': hp.loguniform ('x_reg_alpha', 0., 0.1),
    'reg_lambda': hp.uniform ('x_reg_lambda', 0.9, 1.),
    'n_estimators': hp.quniform ('x_n_estimators', 50, 300, 10)
}

trials = Trials()
best_params = fmin(fn=objective,
            space=space,
            algo=partial(tpe.suggest, n_startup_jobs=1),
            max_evals=100,
            trials=trials)

print("The best params: ", best_params)
X = df_test[features].values

df_test['SalePrice'] = model.predict(X)
df_test[ ['Id','SalePrice'] ].to_csv('../lr.csv', index=False)