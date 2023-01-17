import numpy as np # stats
import pandas as pd # data manipulation
import matplotlib.pyplot as plt # graph
import seaborn as sns # graph
from warnings import filterwarnings

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print('The train set has {} rows, {} columns'.format(*train.shape))
print('The test set has {} rows, {} columns'.format(*test.shape))

all_data = pd.concat([train, test], sort = False, ignore_index = True)
all_data.drop('Id', axis = 1, inplace = True)
print('The all set has {} rows, {} columns'.format(*all_data.shape))
from scipy.stats import norm, skew, probplot, shapiro, normaltest, boxcox
from scipy.special import boxcox1p, inv_boxcox1p

sns.set() # sns default theme
plt.rcParams['figure.figsize'] = [15, 5]
plt.subplots_adjust(wspace = 0.5)
def check_normality(x):
    m, s = norm.fit(x)
    plt.figure()
    plt.subplot(1, 2, 1)
    sns.distplot(x, fit=norm)
    plt.legend(['Normal dist\nμ = {:.2f}\nσ = {:.2f}'.format(m, s)], loc='best', )
    plt.subplot(1, 2, 2)
    probplot(x, plot = plt)
    sk = skew(x)
    sh = shapiro(x)
    print('''
    skew: {:.3f}
    Shapiro-Wilk Test: Statistics = {:.3f}, p = {:.3e}
    D’Agostino’s K^2 Test: Statistics = {:.3f}, p = {:.3e}
    '''.format(skew(x), *shapiro(x), *normaltest(x)))

print(train.SalePrice.describe())
check_normality(train.SalePrice)
check_normality(np.log1p(train.SalePrice))

tmp, maxlog = boxcox(train.SalePrice) # maxlog is the lambda that maximize the log-likelihood function
check_normality(tmp)
all_data = all_data.assign(SalePriceBC = np.append(
    boxcox1p(train.SalePrice, maxlog),
    np.repeat(np.nan, test.shape[0])
))
all_data = all_data.assign(SalePriceL = np.append(
    np.log1p(train.SalePrice),
    np.repeat(np.nan, test.shape[0])
))
all_data.dtypes.value_counts()
tmp = pd.DataFrame({
    'all': all_data.isnull().sum(),
    'train': train.isnull().sum(),
    'test': test.isnull().sum(),
    'dtype': all_data.dtypes,
}, index = all_data.isnull().sum().index) # need this index otherwise the resulting data.frame orders the index by alphabet
tmp
tmp.loc[tmp.iloc[:,0:3].any(1).compress(lambda x: x).index.values]
# a helper function to examine missingness in var with string value
def exam(x, hue = 'Dataset', data = all_data.assign(Dataset = all_data.SalePrice.isnull().replace({ False: 'train', True: 'test' }))):
    data[x].fillna('MISSING', inplace = True)
    plt.figure()
    p = sns.countplot(x = x, hue = hue, data = data)
    for i in p.patches:
        height = i.get_height()
        height = 0 if np.isnan(height) else height
        p.text(i.get_x() + i.get_width() / 2, height + 10, '{:.0f}'.format(height), ha = 'center')
        
# also create a new dataframe so not to mutate the "all"
all1 = all_data.copy()
exam('MSZoning')
all1.MSZoning.fillna('RL', inplace = True)
all1.LotFrontage.describe()
all1.LotFrontage = all1.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
exam('Alley')
all1.Alley.fillna('None', inplace = True)
exam('Utilities')
all1.Utilities.fillna('AllPub', inplace = True)
exam('Exterior1st')
exam('Exterior2nd')

all1[all1.Exterior1st.isnull()]
all1[all1.RoofStyle == 'Flat'].Exterior1st.value_counts()
all1[all1.RoofStyle == 'Flat'].Exterior2nd.value_counts()
all1[all1.RoofMatl == 'Tar&Grv'].Exterior1st.value_counts()
all1[all1.RoofMatl == 'Tar&Grv'].Exterior2nd.value_counts()

all1.Exterior1st.fillna('Plywood', inplace = True)
all1.Exterior2nd.fillna('Plywood', inplace = True)
exam('MasVnrType')
all1.MasVnrType.fillna('None', inplace = True)
all1.MasVnrArea.fillna(0, inplace = True)
exam('BsmtQual')
exam('BsmtCond')
exam('BsmtExposure')
exam('BsmtFinType1')
exam('BsmtFinType2')

# these houses below actually have basement
all1[(
    all1.BsmtQual.isnull() |
    all1.BsmtCond.isnull() |
    all1.BsmtExposure.isnull() | 
    all1.BsmtFinType1.isnull() |
    all1.BsmtFinType2.isnull()
) & all1.TotalBsmtSF > 0]

# change these houses accordingly
all1.loc[all1.BsmtQual.isnull() & all1.TotalBsmtSF > 0, 'BsmtQual'] = 'TA' # TA is the most common
all1.loc[all1.BsmtCond.isnull() & all1.TotalBsmtSF > 0, 'BsmtCond'] = 'TA'
all1.loc[all1.BsmtExposure.isnull() & all1.TotalBsmtSF > 0, 'BsmtExposure'] = 'No' # No is the most common
all1.loc[all1.BsmtFinType2.isnull() & all1.TotalBsmtSF > 0, 'BsmtFinType2'] = 'Unf' # Unf is the most common

# the rest filled with None
all1.BsmtQual.fillna('None', inplace = True)
all1.BsmtCond.fillna('None', inplace = True)
all1.BsmtExposure.fillna('None', inplace = True)
all1.BsmtFinType1.fillna('None', inplace = True)
all1.BsmtFinType2.fillna('None', inplace = True)

# or 0 for numeric
all1.BsmtFinSF1.fillna(0, inplace = True)
all1.BsmtFinSF2.fillna(0, inplace = True)
all1.BsmtUnfSF.fillna(0, inplace = True)
all1.TotalBsmtSF.fillna(0, inplace = True)
all1.BsmtFullBath.fillna(0, inplace = True)
all1.BsmtHalfBath.fillna(0, inplace = True)
exam('Electrical')
all1.Electrical.fillna('SBrkr', inplace = True)
exam('KitchenQual')
all1.KitchenQual.fillna('TA', inplace = True)
exam('Functional')
all1.Functional.fillna('Typ', inplace = True)
exam('FireplaceQu')
all1.FireplaceQu.fillna('None', inplace = True)
all1.GarageYrBlt.describe()

all1.GarageYrBlt.sort_values(ascending = False).index[0] # observation: 2592
all1.loc[2592, ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']]

# change to 2007
all1.loc[2592, 'GarageYrBlt'] = 2007
all1.loc[2592, ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']]

sns.scatterplot(x = 'YearBuilt', y = 'GarageYrBlt', data = all1)
tmp = all1.GarageYrBlt < all1.YearBuilt # those where a garage was built before the house
all1.loc[tmp, ['GarageYrBlt', 'YearBuilt']]
all1.loc[tmp, 'GarageYrBlt'] = all1.loc[tmp, 'YearBuilt']
# all1[all1.GarageType != 'None'].GarageYrBlt.sort_values()
# all1.loc[2217]
exam('GarageType')
exam('GarageFinish')
exam('GarageQual')
exam('GarageCond')

# 2 observations with a garage
all1.loc[all1.GarageType.notnull() & all1.GarageCond.isnull(), 'GarageYrBlt'] = all1.loc[all1.GarageType == 'Detchd', 'GarageYrBlt'].median() # 1962, the median of Detchd type
all1.loc[all1.GarageType.notnull() & all1.GarageCond.isnull(), 'GarageFinish'] = 'Unf' # Unf is the most common
all1.loc[all1.GarageType.notnull() & all1.GarageCond.isnull(), 'GarageQual'] = 'TA' # TA is the most common
all1.loc[all1.GarageType.notnull() & all1.GarageCond.isnull(), 'GarageCond'] = 'Unf' # TA is the most common
all1.loc[all1.GarageType.notnull() & all1.GarageCars.isnull(), 'GarageCars'] = all1.loc[all1.GarageType == 'Detchd', 'GarageCars'].median()
all1.loc[all1.GarageType.notnull() & all1.GarageArea.isnull(), 'GarageArea'] = all1.loc[all1.GarageType == 'Detchd', 'GarageArea'].median()

# the rest
all1.GarageType.fillna('None', inplace = True)
all1.GarageCars.fillna(0, inplace = True)
all1.GarageArea.fillna(0, inplace = True)
all1.GarageFinish.fillna('None', inplace = True)
all1.GarageQual.fillna('None', inplace = True)
all1.GarageCond.fillna('None', inplace = True)

all1.GarageYrBlt = all1.groupby('YearBuilt').GarageYrBlt.transform(lambda x: x.fillna(x.median()))
tmp = all1.GarageYrBlt.isnull()
all1.loc[tmp, 'GarageYrBlt'] = all1.loc[tmp, 'YearBuilt']
exam('PoolQC')
all1.PoolQC.fillna('None', inplace = True)
exam('Fence')
all1.Fence.fillna('None', inplace = True)
exam('MiscFeature')
all1.MiscFeature.fillna('None', inplace = True)
exam('SaleType')
all1.SaleType.fillna('WD', inplace = True)
exam('SaleCondition')
all1[all1.YearBuilt > all1.YrSold] # this house SaleCondition was Partial
all1.loc[:, ~all1.columns.isin(['SalePrice', 'SalePriceBC'])].isnull().any().any()
tmp = all1[all1.SalePrice.notnull()]
tmp_corr = tmp.corr()

plt.figure(figsize = (15, 9))
sns.heatmap(tmp_corr, square = True)

tmp_corr.SalePrice.sort_values(ascending = False)
plt.figure(figsize = (12, 9))
sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = all1)

all1 = all1.assign(IsOutlier = np.where((all1['GrLivArea'] > 4000) & (all1['SalePrice'] < 200000), 1, 0))
# Also check other variables

sns.scatterplot(x = 'LotArea', y = 'SalePrice', data = all1); plt.figure()
sns.scatterplot(x = 'MasVnrArea', y = 'SalePrice', data = all1); plt.figure()
sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice', data = all1); plt.figure()
sns.scatterplot(x = 'GarageArea', y = 'SalePrice', data = all1)
sns.boxplot(x = 'OverallQual', y = 'SalePrice', color = 'seagreen', data = all1)
all1.MSSubClass = all1.MSSubClass.astype(str)
all1.MoSold = all1.MoSold.astype(str)

columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
values = { 'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5 }
all1[columns] = all1[columns].replace(values)

def exam2(x):
    plt.figure()
    sns.boxplot(x = x, y = 'SalePrice', data = all1)

exam2('Street'); all1.Street.replace({ 'Grvl': 0, 'Pave': 1 }, inplace = True)
exam2('Alley'); all1.Alley.replace({ 'No': 0, 'Grvl': 1, 'Pave': 2 }, inplace = True)

# these don't seem ordinal
# exam2('LotShape'); exam2('LandContour'); exam2('LotConfig'); exam2('LandSlope')
# exam2('Condition1'); exam2('Condition2')
# exam2('BldgType'); exam2('HouseStyle')
# exam2('RoofStyle'); exam2('RoofMatl'); exam2('MasVnrType'); exam2('Foundation')
# exam2('Heating')

exam2('BsmtExposure'); all1.BsmtExposure.replace({ 'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4 }, inplace = True)
values = { 'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6 }
exam2('BsmtFinType1'); all1.BsmtFinType1.replace(values, inplace = True)
exam2('BsmtFinType2'); all1.BsmtFinType2.replace(values, inplace = True)

# exam2('Heating') # not ordinal
exam2('CentralAir'); all1.CentralAir.replace({ 'Y': 1, 'N': 0 }, inplace = True)

# exam2('Electrical') # not ordinal

exam2('Functional'); all1.Functional.replace({ 'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7 }, inplace = True)

# exam2('GarageType') # not ordinal
exam2('GarageFinish'); all1.GarageFinish.replace({ 'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3 }, inplace = True)
exam2('PavedDrive'); all1.PavedDrive.replace({ 'N': 0, 'P': 1, 'Y': 2 }, inplace = True)

# exam2('MiscFeature'); exam2('SaleType'); exam2('SaleCondition')
# GrLivArea is usually the sum of 1stFlrSF and 2ndFlrSF, except a few 
tmp = all1[['1stFlrSF', '2ndFlrSF']].apply(sum, axis = 1) == all1.GrLivArea
all1.loc[tmp == 0, ['GrLivArea', '1stFlrSF', '2ndFlrSF']]

# Assign TotalArea
all1 = all1.assign(TotalArea = all1[['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'MasVnrArea']].apply(sum, axis = 1))
all1[['TotalArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', 'MasVnrArea', 'SalePrice']].corr()
all1 = all1.assign(Bath = all1.BsmtFullBath + all1.BsmtHalfBath / 2 + all1.FullBath + all1.HalfBath / 2)
all1[['Bath', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'SalePrice']].corr()
all1 = all1.assign(Age = all1.YrSold - all1.YearBuilt)
all1.loc[all1.Age < 0, 'Age'] = 0 # There was one house where YrSold < YearBuilt

all1 = all1.assign(IsRemodelled = np.where(all1.YearBuilt == all1.YearRemodAdd, 0, 1))

# all1 = all1.assign(IsNew = np.where(all1.YrSold <= all1.YearBuilt, 1, 0)) # use <= because there was one house where YrSold < YearBuilt
# all1.loc[all1.SaleType == 'New', 'IsNew'] = 1 # some where 

all1.Age.describe()
sns.distplot(all1.Age, bins = 50); plt.figure()
sns.scatterplot(x = 'Age', y = 'SalePrice', data = all1); plt.figure()
sns.boxplot(x = 'IsRemodelled', y = 'SalePrice', data = all1); plt.figure()
sns.boxplot(x = 'SaleType', y = 'SalePrice', data = all1); plt.figure()
filterwarnings('ignore')
tmp = all1.skew().compress(lambda x: abs(x) > 0.75).sort_values()
tmp = tmp[tmp.index != 'MSSubClass']
tmp = pd.DataFrame({
    'ori': tmp,
    'after with log': [skew(np.log1p(all1[x]), nan_policy = 'omit') for x in tmp.index],
    'after with boxcox': [skew(boxcox(all1[x] + 1)[0], nan_policy = 'omit') for x in tmp.index]
})
filterwarnings('ignore')
tmp
for v in ['Functional', 'BsmtCond', 'GarageQual', 'PavedDrive', 'BsmtQual', 'TotRmsAbvGrd', 'ExterQual', '2ndFlrSF', 'BsmtUnfSF', 'BsmtExposure', 'TotalBsmtSF', 'ExterCond', 'BsmtFinSF1', 'TotalArea', 'LotFrontage', 'WoodDeckSF', 'OpenPorchSF', 'MasVnrArea', 'BsmtFinType2', 'ScreenPorch', 'EnclosedPorch', 'BsmtFinSF2', 'KitchenAbvGr', '3SsnPorch', 'LowQualFinSF', 'PoolArea', 'PoolQC', 'MiscVal']:
    all1[v] = boxcox(all1[v] + 1)[0]

for v in ['GrLivArea', '1stFlrSF', 'LotArea']:
    all1[v] = np.log1p(all1[v])
all1.skew().compress(lambda x: abs(x) > 0.75).sort_values()
all1.dtypes.value_counts()
from category_encoders import OneHotEncoder, BinaryEncoder
# all1_n = BinaryEncoder().fit_transform(all1) # tried this, but cv is lower than onehot
all1_n = pd.get_dummies(all1) # similar to all1_n = OneHotEncoder().fit_transform(all1)

print(all1_n.shape)
print(all1_n.dtypes.value_counts())

train_index = all1_n.SalePrice.notnull()

# remove the outliers
outliers = all1_n.IsOutlier == 1
X = all1_n.loc[~outliers & train_index, ~all1_n.columns.isin(['SalePrice', 'SalePriceBC', 'SalePriceL', 'IsOutlier'])]
y = all1_n.loc[~outliers & train_index, 'SalePriceBC'] # boxcox transformed SalePrice
yL = all1_n.loc[~outliers & train_index, 'SalePriceL'] # log transformed SalePrice
yO = all1_n.loc[~outliers & train_index, 'SalePrice'] # Original SalePrice

X_test = all1_n.loc[~train_index, ~all1_n.columns.isin(['SalePrice', 'SalePriceBC', 'SalePriceL', 'IsOutlier'])]
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error

# score model performance
cv = KFold(n_splits = 5, shuffle = True, random_state = 0)
def score(model, X = X, y = y, des = ''):
    rmse = np.sqrt(-cross_val_score(model, X = X, y = y, scoring = 'neg_mean_squared_error', cv = cv))
    print(des, '\n', *rmse, '\nMean:', rmse.mean(), '\nSD:', rmse.std())
    return(rmse)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, LassoLarsIC, ElasticNet, ElasticNetCV, BayesianRidge
# baseline, if we guess all y as y.mean()
# np.sqrt(mean_squared_error(y, np.repeat(y.mean(), len(y)))) # boxcox
# np.sqrt(mean_squared_error(yL, np.repeat(yL.mean(), len(yO)))) # log
# np.sqrt(mean_squared_error(yO, np.repeat(yO.mean(), len(yL)))) # original
model = make_pipeline(RobustScaler(), LinearRegression())
score(model, des = 'Using boxcox transformed SalePrice as y');
score(model, y = yL, des = '\nUsing log transformed SalePrice as y');
# score(model, y = yO, des = '\nUsing original SalePrice as y');
filterwarnings('ignore')

model = make_pipeline(RobustScaler(), Ridge())
score(model, des = 'Using boxcox transformed SalePrice as y');
score(model, y = yL, des = '\nUsing log transformed SalePrice as y');
# score(model, y = yO, des = '\nUsing original SalePrice as y');

filterwarnings('default')
filterwarnings('ignore')

score(make_pipeline(Ridge()), des = 'No scaling'); # without scaling seems a little better
score(make_pipeline(StandardScaler(), Ridge()), des = '\nUsing StandardScaler'); # standard scaler seems a litte bit worse

filterwarnings('default')
# Using LassoCV to find the best alpha
filterwarnings('ignore')
model = LassoCV(
    alphas = np.logspace(-5, 0.5, 50),
    random_state = 0,
    cv = cv,
    max_iter = 100000,
    n_jobs = -1
)
model.fit(RobustScaler().fit_transform(X), y)
filterwarnings('default')

alpha = model.alpha_

def plot_rmse(alphas, rmse):    
    tmp = pd.DataFrame({
        'alpha': alphas,
        '-log(alpha)': -np.log(alphas),
        'rmse': rmse,
    })
    sns.scatterplot(x = '-log(alpha)', y = 'rmse', data = tmp)
    tmp = tmp[tmp.rmse == min(tmp.rmse)] # the min
    plt.annotate(
        'alpha: {:.6f}\nscore: {:.6f}'.format(*tmp[['alpha', 'rmse']].iloc[0]),
        xy = (tmp['-log(alpha)'], tmp['rmse'] + 0.001),
        xytext = (7.5, tmp['rmse'] + 0.025),
        arrowprops = dict(arrowstyle = '->', color = 'green')
    )

plot_rmse(model.alphas_, np.sqrt(np.mean(model.mse_path_, 1)))
tmp = pd.DataFrame({
    'coef': model.coef_,
    'var': X.columns.values
})
tmp = tmp.drop(tmp[abs(tmp.coef) < 0.01].index).sort_values('coef')
tmp
# Using GridSearchCV to find the best alpha
from sklearn.model_selection import GridSearchCV

filterwarnings('ignore')
model = GridSearchCV(
    Lasso(random_state = 0),
    param_grid = { 'alpha': np.logspace(-5, 0, 50) },
    scoring = 'neg_mean_squared_error',
    n_jobs = -1,
    return_train_score = True,
    cv = cv,
)
model.fit(RobustScaler().fit_transform(X), y)
filterwarnings('default')

for i in model.cv_results_: print(i)
plot_rmse(
    model.cv_results_['param_alpha'].data.astype(np.float64),
    np.sqrt(-model.cv_results_['mean_test_score'])
)

alpha = model.best_params_['alpha']
model = make_pipeline(RobustScaler(), Lasso(alpha = 0.000212, random_state = 0, max_iter = 10000))
score(model, des = 'Using boxcox transformed SalePrice as y');
score(model, y = yL, des = '\nUsing log transformed SalePrice as y');
# score(model, y = yO, des = '\nUsing original SalePrice as y'); # doesn't converge
# LassoLarsIC
filterwarnings('ignore')
model = make_pipeline(RobustScaler(), LassoLarsIC())
score(model, des = 'Using boxcox transformed SalePrice as y');
score(model, y = yL, des = '\nUsing log transformed SalePrice as y');
# score(model, y = yO, des = '\nUsing original SalePrice as y');
filterwarnings('default')
# test the effects of NOT excluding the outliers
model = make_pipeline(RobustScaler(), Lasso(alpha = 0.000212, random_state = 0, max_iter = 100000))
score(
    model,
    X = all1_n.loc[train_index, ~all1_n.columns.isin(['SalePrice', 'SalePriceBC', 'SalePriceL', 'IsOutlier'])],
    y = all1_n.loc[train_index, 'SalePriceBC']
);
# # Using ElasticNetCV to find the best alpha and l1_ratio
model = ElasticNetCV(
    random_state = 0, cv = cv, max_iter = 100000, n_jobs = -1,
    alphas = np.logspace(-4, 0, 25),
    l1_ratio = [0.1, .5, .7, .9, .95, .99]
)
filterwarnings('ignore')
model.fit(X, y)
filterwarnings('default')

print('Best alpha: {}\nbest l1_ratio: {}'.format(model.alpha_, model.l1_ratio_))
model = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.001695, l1_ratio = 0.1, random_state = 0, max_iter = 100000))
score(model, des = 'Using boxcox transformed SalePrice as y');
score(model, y = yL, des = '\nUsing log transformed SalePrice as y');
pd.DataFrame({
    'Id': test.Id,
    'SalePrice': inv_boxcox1p(model.fit(X, y).predict(X_test), maxlog)
}).to_csv('submission_elastic_net_y.csv', index = False)

pd.DataFrame({
    'Id': test.Id,
    'SalePrice': np.expm1(model.fit(X, yL).predict(X_test))
}).to_csv('submission_elastic_net_yL.csv', index = False)
# BayesianRidge
model = make_pipeline(RobustScaler(), BayesianRidge())
score(model, des = 'Using boxcox transformed SalePrice as y');
score(model, y = yL, des = '\nUsing log transformed SalePrice as y');
from sklearn.ensemble import RandomForestRegressor
model = make_pipeline(RandomForestRegressor(n_estimators = 500, n_jobs = -1))
filterwarnings('ignore')
score(model);
filterwarnings('default')
from sklearn.ensemble import GradientBoostingRegressor
model = make_pipeline(GradientBoostingRegressor())
filterwarnings('ignore')
score(model);
score(model, y = yL);
filterwarnings('default')
# from xgboost import XGBRegressor
from xgboost.sklearn import XGBRegressor # with scikit-learn wrapper
model = XGBRegressor(
    learning_rate = 0.046777,
    n_estimators = 250,
    max_depth = 4,
    min_child_weight = 1,
    gamma = 0,
    subsample = 0.5,
    colsample_bytree = 0.6,
    reg_alpha = 0.17704,
    reg_lambda = 0.75,
    n_jobs = -1,
)
score(model);
score(model, y = yL);
pd.DataFrame({
    'Id': test.Id,
    'SalePrice': inv_boxcox1p(model.fit(X, y).predict(X_test), maxlog)
}).to_csv('submission_xgb_y.csv', index = False)

pd.DataFrame({
    'Id': test.Id,
    'SalePrice': np.expm1(model.fit(X, yL).predict(X_test))
}).to_csv('submission_xgb_yL.csv', index = False)
# tuning XGBRegressor
# # 1, tune n_estimators, best value found: 250, 0.047966
# model = GridSearchCV(
#     XGBRegressor(n_jobs = -1),
#     { 'n_estimators': np.arange(100, 500, 50) },
#     scoring = 'neg_mean_squared_error', return_train_score = True, cv = cv, n_jobs = -1,
# )

# # 2, tune max_depth, best value found: 4, 0.047581
# model = GridSearchCV(
#     XGBRegressor(n_jobs = -1, n_estimators = 250),
#     { 'max_depth': np.arange(1, 9, 1) },
#     scoring = 'neg_mean_squared_error',
#     return_train_score = True,
#     cv = cv,
#     n_jobs = -1,
# )
# # 3, tune min_child_weight, best value found: 2, 0.047548
# model = GridSearchCV(
#     XGBRegressor(n_jobs = -1, n_estimators = 250, max_depth = 4),
#     { 'min_child_weight': np.arange(1, 9, 1) },
#     scoring = 'neg_mean_squared_error',
#     return_train_score = True,
#     cv = cv,
#     n_jobs = -1,
# )
# # 4, tune gamma, best value found: 0, 0.047581
# model = GridSearchCV(
#     XGBRegressor(n_jobs = -1, n_estimators = 250, max_depth = 4, min_child_weight = 1),
#     { 'gamma': np.arange(0, 0.5, 0.1) },
#     scoring = 'neg_mean_squared_error',
#     return_train_score = True,
#     cv = cv,
#     n_jobs = -1,
# )
# # 5, tune subsample & colsample_bytree, best value found: 0.5 & 0.6,  0, 0.046886
# model = GridSearchCV(
#     XGBRegressor(n_jobs = -1, n_estimators = 250, max_depth = 4, min_child_weight = 1, gamma = 0, ),
#     { 'subsample': np.arange(0.5, 1.01,  0.1), 'colsample_bytree': np.arange(0.5, 1.01, 0.1) },
#     scoring = 'neg_mean_squared_error',
#     return_train_score = True,
#     cv = cv,
#     n_jobs = -1,
# )
# # 6, tune alpha & lambda, best value found: 0.17704 & 0.75, 0.046783
# model = GridSearchCV(
#     XGBRegressor(n_jobs = -1, n_estimators = 250, max_depth = 4, min_child_weight = 1, gamma = 0, subsample = 0.5, colsample_bytree = 0.6),
#     { 'reg_alpha': np.logspace(np.log(1e-4), np.log(1.5), base = np.exp(1), num = 10), 'reg_lambda': [.1, .25, .5, .75, .9, .95, .99, 1] },
#     scoring = 'neg_mean_squared_error',
#     return_train_score = True,
#     cv = cv,
#     n_jobs = -1,
# )
# # 7, tune learning_rate, best value found: 0.1, 0.046777
# model = GridSearchCV(
#     XGBRegressor(n_jobs = -1, n_estimators = 250, max_depth = 4, min_child_weight = 1, gamma = 0, subsample = 0.5, colsample_bytree = 0.6, reg_alpha = 0.17704, reg_lambda = 0.75),
#     { 'learning_rate': [0.01, 0.025, 0.05, 0.75, 0.1] },
#     scoring = 'neg_mean_squared_error',
#     return_train_score = True,
#     cv = cv,
#     n_jobs = -1,
# )
# the script below run a bayesian optimization search for parameters, take many hours

# from timeit import default_timer as timer
# import pickle

# # https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0
# from hyperopt import hp, fmin, Trials, tpe, STATUS_OK

# # trials = Trials()
# # or
# # with open('xgb_trials.pickle', 'rb') as handle: trials = pickle.load(handle)

# def objective(params):
#     for k in params: params[k] = [params[k]]
#     params['n_estimators'] = [int(params['n_estimators'][0])]
#     params['max_depth'] = [int(params['max_depth'][0])]
#     params['min_child_weight'] = [int(params['min_child_weight'][0])]    
#     model = GridSearchCV(
#         XGBRegressor(n_jobs = -1),
#         params,
#         scoring = 'neg_mean_squared_error',
#         return_train_score = True,
#         cv = cv,
#         n_jobs = -1,
#     )
#     model.fit(X, y) 
#     result = {
#         'loss': np.sqrt(-model.best_score_),
#         'params': model.best_params_,
#         'elapsed': timer() - start_time,
#     }
#     rounded_result = { k: round(v, 6) if type(v) != dict else { k2: round(v2, 6) for k2, v2, in v.items() } for k, v in result.items() }
#     n_trial = len(trials.tids)
#     print('Trial', n_trial, rounded_result)
#     # save the trials every now and then
#     if n_trial % 20 == 0:
#         with open('xgb_trials.pickle', 'wb') as handle:
#             pickle.dump(trials, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
#     result['status'] = STATUS_OK
#     return(result)

# # set up timer & run
# start_time = timer()

# best = fmin(
#     fn = objective,
#     space = {
#         'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
#         'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
#         'max_depth': hp.quniform('max_depth', 3, 10, 1),
#         'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
#         'gamma': hp.uniform('gamma', 0, 0.4),
#         'subsample': hp.uniform('subsample', 0.6, 0.9),
#         'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 0.9),
#         'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(1)),
#     },
#     algo = tpe.suggest,
#     trials = trials,
#     rstate = np.random.RandomState(0),
#     max_evals = 10000,
# )
from lightgbm import LGBMRegressor
model = make_pipeline(LGBMRegressor())
score(model);