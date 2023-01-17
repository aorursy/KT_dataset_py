import numpy as np

import pandas as pd

import matplotlib as plt

import math

import re

from operator import itemgetter



from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import ExtraTreesRegressor

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
sample_submission = pd.read_csv('../input/sample_submission.csv')

raw_df = pd.read_csv('../input/train.csv')

raw_test = pd.read_csv('../input/test.csv')
def display_df(df):

    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000):

        display(df)

        

def rmse(x,y): 

    return math.sqrt(((x-y)**2).mean())



def print_score(clf, X_train, y_train, X_val, y_val):

    res = [rmse(clf.predict(X_train), y_train), rmse(clf.predict(X_val), y_val),

                clf.score(X_train, y_train), clf.score(X_val, y_val)]

    if hasattr(clf, 'oob_score_'): res.append(clf.oob_score_)

    print(res)
display_df(raw_df.describe(include='all').T)
# Convert sale price to log of sale price

df = raw_df.copy()

df['SalePrice'] = np.log(raw_df['SalePrice'])
def split_sets(df, n, shuffle=True):

    if shuffle:

        df = df.sample(frac=1).reset_index(drop=True)

        

    return df[:n], df[n:]



def convert_strings_to_categorical(train, val, key, max_cat=0):

    if pd.api.types.is_string_dtype(train[key]):

        train[key] = train[key].astype('category').cat.as_ordered()

        

        if key in val.columns:

            val[key] = val[key].astype('category').cat.as_ordered()

            val[key].cat.set_categories(train[key].cat.categories, ordered=True, inplace=True)

        

        if len(train[key].cat.categories) > max_cat:

            train[key] = pd.Categorical(train[key]).codes

            val[key] = pd.Categorical(val[key]).codes

            

def fix_missing_years(train, val, col):

    if 'Yr' in col or 'Year' in col:

        train[col] = train[col].fillna(0)

        val[col] = val[col].fillna(0)



def add_median_to_missing_values(train, val, key, testing=True):

    if pd.api.types.is_numeric_dtype(train[key]):

        median = train[key].median()

        

        if any(train[key].isna()):

            print('Filling NAs in train columns {} with median: {}'.format(key, median))

            train[key] = train[key].fillna(median)

        

        if key in val.columns:

            if any(val[key].isna()):

                print('Filling NAs in test column {} with median: {}'.format(key, median))

                val[key] = val[key].fillna(median)

        else:

            if testing:

                print('Column which is not in train was found in validation set. Removing: {}'.format(key))

                val.drop(key, axis=1, inplace=True)

            

def _get_dummies(train, val, dummy_columns=None):

    if dummy_columns:

        train = pd.get_dummies(train, columns=dummy_columns)

        val = pd.get_dummies(val, columns=dummy_columns)

    else:

        train = pd.get_dummies(train)

        val = pd.get_dummies(val)

    

    train, val = train.align(val, 'left', axis=1, fill_value=0)

    assert set(train.columns) == set(val.columns)

    

    return train, val
# Split data into train and validation sets

train, val = split_sets(df.copy(), 1000, shuffle=True)



# Convert features to numerical

for key, value in train.items():

    fix_missing_years(train, val, key)

    add_median_to_missing_values(train, val, key)

    convert_strings_to_categorical(train, val, key)

    

train, val = _get_dummies(train, val)



# Split into train and validation set reflecting x and y 

X_train = train.drop('SalePrice', axis=1)

y_train = train['SalePrice']



X_val = val.drop('SalePrice', axis=1)

y_val = val['SalePrice']



# Quick random forest

clf = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, max_depth=None, max_features='sqrt', n_jobs=-1)

clf.fit(X_train, y_train)



print('Shape of training set: {}'.format(X_train.shape))

print_score(clf, X_train, y_train, X_val, y_val)
def get_feature_imp(clf, X_train):

    with open('../input/data_description.txt', 'r') as f:

        desc = f.read()



    fd = {}

    for n in desc.split('\n'):

        if ':' in n:

            fd[n.split(':')[0].strip()] = n.split(':')[1].strip()



    feature_desc = []

    for x in X_train.columns:

        x = re.sub(r'_.*', '', x)

        if x in fd.keys():

            feature_desc.append(fd[x])

        else:

            feature_desc.append(None)



    # sorted(list(zip(X_train.columns, clf.feature_importances_, feature_desc)), key=itemgetter(1), reverse=True)

    top_f = pd.DataFrame({'features': X_train.columns, 'score': clf.feature_importances_, 'description': feature_desc})

    return top_f



top_f = get_feature_imp(clf, X_train)

top_f_sorted = top_f.sort_values('score', ascending=False)

display_df(top_f_sorted)
to_keep = list(top_f_sorted[top_f_sorted['score'] >= 0.005]['features'])

print('will keep {} features'.format(len(to_keep)))



clf = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, max_depth=None, max_features='sqrt', n_jobs=-1)

clf.fit(X_train[to_keep], y_train)



print('Shape of training set: {}'.format(X_train[to_keep].shape))

print_score(clf, X_train[to_keep], y_train, X_val[to_keep], y_val)
df = raw_df.copy()



def get_feature_desc(df):

    with open('../input/data_description.txt', 'r') as f:

        desc = f.read()



    fd = {}

    for n in desc.split('\n'):

        if ':' in n:

            fd[n.split(':')[0].strip()] = n.split(':')[1].strip()



    feature_desc = []

    for x in df.columns:

        x = re.sub(r'_.*', '', x)

        if x in fd.keys():

            feature_desc.append(fd[x])

        else:

            feature_desc.append(None)



    # sorted(list(zip(X_train.columns, clf.feature_importances_, feature_desc)), key=itemgetter(1), reverse=True)

    f_desc = pd.DataFrame({'features': df.columns, 'description': feature_desc, '% missing': df.isnull().sum()/len(df)*100})

    return f_desc



display_df(get_feature_desc(df))
def fix_values(df):

    df['Alley'] = df['Alley'].fillna('None')

    df['Exterior2nd'] = df['Exterior2nd'].fillna('None')

    df['GarageType'] = df['GarageType'].fillna('None')

    df['MiscFeature'] = df['MiscFeature'].fillna('None')

    df['MasVnrType'] = df['MasVnrType'].fillna('Unknown')

    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    df['LotFrontage'] = df['LotFrontage'].fillna(0)

    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

    df['LotShape'] = pd.Categorical(df['LotShape'], ordered=True, categories=['IR3', 'IR2', 'IR1', 'Reg'])

    df['LandContour'] = pd.Categorical(df['LandContour'], ordered=True, categories=['Low', 'HLS', 'BNK', 'Lvl'])

    df['Utilities'] = pd.Categorical(df['Utilities'], ordered=True, categories=['ELO', 'NoSeWa', 'NoSewr', 'AllPub'])

    df['LandSlope'] = pd.Categorical(df['LandSlope'], ordered=True, categories=['Sev', 'Mod', 'Gnt'])

    df['OverallQual'] = pd.Categorical(df['OverallQual'], ordered=True)

    df['OverallCond'] = pd.Categorical(df['OverallCond'], ordered=True)



    df['ExterQual'] = df['ExterQual'].fillna('TA')

    df['ExterQual'] = pd.Categorical(df['ExterQual'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['ExterCond'] = df['ExterCond'].fillna('TA')

    df['ExterCond'] = pd.Categorical(df['ExterCond'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['BsmtQual'] = df['BsmtQual'].fillna('None')

    df['BsmtQual'] = pd.Categorical(df['BsmtQual'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['BsmtCond'] = df['BsmtCond'].fillna('None')

    df['BsmtCond'] = pd.Categorical(df['BsmtCond'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['BsmtExposure'] = df['BsmtExposure'].fillna('None')

    df['BsmtExposure'] = pd.Categorical(df['BsmtExposure'], ordered=True, categories=['None', 'No', 'Mn', 'Av', 'Gd'])



    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('None')

    df['BsmtFinType1'] = pd.Categorical(df['BsmtFinType1'], ordered=True, categories=['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])



    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('None')

    df['BsmtFinType2'] = pd.Categorical(df['BsmtFinType2'], ordered=True, categories=['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])



    df['HeatingQC'] = df['HeatingQC'].fillna('TA')

    df['HeatingQC'] = pd.Categorical(df['HeatingQC'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['KitchenQual'] = df['KitchenQual'].fillna('TA')

    df['KitchenQual'] = pd.Categorical(df['KitchenQual'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['Functional'] = df['Functional'].fillna('Unknown')

    df['Functional'] = pd.Categorical(df['Functional'], ordered=True, categories=['Unknown', 'Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'])



    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

    df['FireplaceQu'] = pd.Categorical(df['FireplaceQu'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['GarageFinish'] = df['GarageFinish'].fillna('None')

    df['GarageFinish'] = pd.Categorical(df['GarageFinish'], ordered=True, categories=['None', 'Unf', 'RFn', 'Fin'])



    df['GarageQual'] = df['GarageQual'].fillna('None')

    df['GarageQual'] = pd.Categorical(df['GarageQual'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['GarageCond'] = df['GarageCond'].fillna('None')

    df['GarageCond'] = pd.Categorical(df['GarageCond'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['PavedDrive'] = df['PavedDrive'].fillna('None')

    df['PavedDrive'] = pd.Categorical(df['PavedDrive'], ordered=True, categories=['None', 'N', 'P', 'Y'])



    df['PoolQC'] = df['PoolQC'].fillna('None')

    df['PoolQC'] = pd.Categorical(df['PoolQC'], ordered=True, categories=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])



    df['Fence'] = df['Fence'].fillna('None')

    df['Fence'] = pd.Categorical(df['Fence'], ordered=True, categories=['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'])



    df['TotAreaInside'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GrLivArea'] + df['GarageArea']

    df['TotAreaOutside'] = df['LotArea'] + df['PoolArea']

    

    return df



one_hot = ['MSZoning', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',

          'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'MiscFeature', 'SaleType', 'SaleCondition',

          'Street', 'CentralAir', 'GarageType']

area_inside = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea']

area_outside = ['LotArea', 'PoolArea']
df = fix_values(raw_df.copy())

df['SalePrice'] = np.log(raw_df['SalePrice'])

df['LandSlope'] = df.groupby('Neighborhood')['LandSlope'].transform(lambda x: x.fillna(x.value_counts().index[0]))

df['LandContour'] = df.groupby('Neighborhood')['LandContour'].transform(lambda x: x.fillna(x.value_counts().index[0]))

df['Electrical'] = df.groupby('Neighborhood')['Electrical'].transform(lambda x: x.fillna(x.value_counts().index[0]))
plt.figure(figsize=(15, 5))

plt.subplot(121)

plt.scatter(df['YearRemodAdd'], df['SalePrice'])



def convert_remodelled_to_boolean(df, drop=True):

    df['was_remodelled'] = 1

    for index, entry in df.iterrows():

        if entry['YearRemodAdd'] == 1950 or entry['YearRemodAdd'] == entry['YearBuilt']:

            df.loc[index, 'was_remodelled'] = 0

            

    if drop:

        df.drop('YearRemodAdd', axis=1, inplace=True)

    

    return df



df = convert_remodelled_to_boolean(df.copy())



plt.subplot(122)

plt.scatter(df['was_remodelled'], df['SalePrice'])

plt.show()
df.head()
def apply_cats(val, train):

    for n, c in val.items():

        if (n in train.columns) and (train[n].dtype.name=='category'):

            val[n] = c.astype('category').cat.as_ordered()

            val[n].cat.set_categories(train[n].cat.categories, ordered=True, inplace=True)

            

            train[n] = pd.Categorical(train[n]).codes

            val[n] = pd.Categorical(val[n]).codes
# Split data into train and validation sets

train, val = split_sets(df.copy(), 1000, shuffle=True)



apply_cats(val, train)



train, val = _get_dummies(train, val, dummy_columns=one_hot)



# Split into train and validation set reflecting x and y 

X_train = train.drop('SalePrice', axis=1)

y_train = train['SalePrice']



X_val = val.drop('SalePrice', axis=1)

y_val = val['SalePrice']



# Quick random forest

clf = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, max_depth=None, max_features='sqrt', n_jobs=-1)

clf.fit(X_train, y_train)



print('Shape of training set: {}'.format(X_train.shape))

print_score(clf, X_train, y_train, X_val, y_val)
top_f = get_feature_imp(clf, X_train)

top_f_sorted = top_f.sort_values('score', ascending=False)

display_df(top_f_sorted)
top_n = 20

cols = list(top_f_sorted[:top_n]['features'])

f = plt.figure(figsize=(20, 40))

for index, col in enumerate(train[cols].columns):

    plt.subplot(math.ceil(len(train[cols].columns)/3), 3, index+1)

    plt.scatter(train[col], train['SalePrice'])

    plt.title(top_f[top_f['features'] == col]['description'].item())

    plt.xlabel(col)

    plt.ylabel('SalePrice')



plt.show()
# Let's remove outlies from each of those top categories

def remove_outliers(df):

    # df = df[~((df['OverallQual'] > 8) & (df['SalePrice'] < 12.2))]

    df = df[df['TotAreaInside'] < 12500]

#     df = df[df['TotalBsmtSF'] < 4000]

#     df = df[df['GarageArea'] < 1200]

    df = df[df['GrLivArea'] < 4000]

#     df = df[df['1stFlrSF'] < 3000]

#     df = df[df['GarageCars'] < 3.5]

    df = df[df['LotArea'] < 60000]

#     df = df[df['BsmtFinSF1'] < 4000]

#     df = df[df['TotAreaOutside'] < 80000]

#     df = df[df['Fireplaces'] < 2.5]

    

    return df
def fix_skewness(df):

    numerics2 = []

    for i in df.columns:

        if pd.api.types.is_numeric_dtype(df[i]):

            numerics2.append(i)

            

    skew_features = df[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



    high_skew = skew_features[skew_features > 0.5]

    skew_index = high_skew.index



    for i in skew_index:

        df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))

        

    return df
df = fix_values(raw_df.copy())

df['SalePrice'] = np.log(raw_df['SalePrice'])

df['LandSlope'] = df.groupby('Neighborhood')['LandSlope'].transform(lambda x: x.fillna(x.value_counts().index[0]))

df['LandContour'] = df.groupby('Neighborhood')['LandContour'].transform(lambda x: x.fillna(x.value_counts().index[0]))

df['Electrical'] = df.groupby('Neighborhood')['Electrical'].transform(lambda x: x.fillna(x.value_counts().index[0]))

df = convert_remodelled_to_boolean(df.copy())

#df = fix_skewness(df)



# Split data into train and validation sets

train, val = split_sets(df.copy(), 1100, shuffle=True)



apply_cats(val, train)



train, val = _get_dummies(train, val, dummy_columns=one_hot)

#train = remove_outliers(train)



# Split into train and validation set reflecting x and y 

X_train = train.drop('SalePrice', axis=1)

y_train = train['SalePrice']



X_val = val.drop('SalePrice', axis=1)

y_val = val['SalePrice']



# Quick random forest

clf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_depth=8, max_features='sqrt', n_jobs=-1)

clf.fit(X_train, y_train)



print('Shape of training set: {}'.format(X_train.shape))

print_score(clf, X_train, y_train, X_val, y_val)
class lr_ensemble():

    def __init__(self, models):

        self.models = models



    def train(self, X, y):

        predictions = []

        

        for model in models.values():

            preds = model.predict(X)

            predictions.append(preds)

            

        x_train = np.reshape(np.array(predictions).T, (len(y), len(models)), order='F')

        self.lr = LinearRegression()

        self.lr.fit(x_train, y)



    def predict(self, X, method='lr'):

        predictions = []

        

        for model in models.values():

            preds = model.predict(X)

            predictions.append(preds)

            

        x_train = np.reshape(np.array(predictions).T, (len(X), len(models)), order='F')



        if method == 'lr':

            return self.lr.predict(x_train)

        else:

            return [x.mean() for x in x_train]

        

    

    def score(self, X, y):

        _X = self.predict(X)

        u = ((y - _X) ** 2).sum()

        v = ((y - y.mean()) ** 2).sum()

        return (1 - u/v)
models = {

    'random_forest': RandomForestRegressor(n_estimators=1000, min_samples_leaf=3, max_depth=6, max_features='sqrt', n_jobs=-1),

    'extra_trees': ExtraTreesRegressor(n_estimators=2000, max_depth=8, min_samples_leaf=3, max_features='sqrt', n_jobs=-1),

    'ada_boost': AdaBoostRegressor(n_estimators=2000, learning_rate=.1),

    'xgboost': GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, 

                                         min_samples_split=10, loss='huber'),

    'lgbm': LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75,

                          bagging_freq=5, bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7, verbose=-1),

    'gbr': XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.7,

                        bjective='reg:linear', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=0.00006)

}
for name, model in models.items():

    print('Training {}'.format(name))

    model.fit(X_train, y_train)
ensemble = lr_ensemble(models)

ensemble.train(X_train, y_train)
def print_score2(clf, X_train, y_train, X_val, y_val):

    res = [rmse(clf.predict(X_train, method='mean'), y_train), rmse(clf.predict(X_val, method='mean'), y_val),

                clf.score(X_train, y_train), clf.score(X_val, y_val)]

    if hasattr(clf, 'oob_score_'): res.append(clf.oob_score_)

    print(res)



print_score2(ensemble, X_train, y_train, X_val, y_val)
train = raw_df.copy()

test = raw_test.copy()



y_train = train['SalePrice']

train = train.drop('SalePrice', axis=1)



df = pd.concat((train, test))

df = fix_values(df)

# df['SalePrice'] = np.log(raw_df['SalePrice'])

df['LandSlope'] = df.groupby('Neighborhood')['LandSlope'].transform(lambda x: x.fillna(x.value_counts().index[0]))

df['LandContour'] = df.groupby('Neighborhood')['LandContour'].transform(lambda x: x.fillna(x.value_counts().index[0]))

df['Electrical'] = df.groupby('Neighborhood')['Electrical'].transform(lambda x: x.fillna(x.value_counts().index[0]))

df = convert_remodelled_to_boolean(df.copy())



train = df[:len(train)]

test = df[len(train):]



apply_cats(test, train)



train, test = _get_dummies(train, test, dummy_columns=one_hot)
# models = {

#     'random_forest': RandomForestRegressor(n_estimators=120, min_samples_leaf=3, max_depth=6, max_features='sqrt', n_jobs=-1),

#     'extra_trees': ExtraTreesRegressor(n_estimators=180, max_depth=8, min_samples_leaf=3, max_features='sqrt', n_jobs=-1),

#     'ada_boost': AdaBoostRegressor(n_estimators=100, learning_rate=.1),

#     'xgboost': GradientBoostingRegressor()

# }



models = {

    'random_forest': RandomForestRegressor(n_estimators=1000, min_samples_leaf=3, max_depth=6, max_features='sqrt', n_jobs=-1),

    'extra_trees': ExtraTreesRegressor(n_estimators=2000, max_depth=8, min_samples_leaf=3, max_features='sqrt', n_jobs=-1),

    'ada_boost': AdaBoostRegressor(n_estimators=2000, learning_rate=.1),

    'xgboost': GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, 

                                         min_samples_split=10, loss='huber'),

    'lgbm': LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75,

                          bagging_freq=5, bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7, verbose=-1),

    'gbr': XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.7,

                        bjective='reg:linear', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=0.00006)

}



for model in models.values():

    model.fit(train, y_train)
ensemble = lr_ensemble(models)

ensemble.train(train, y_train)
test = test.fillna(0)
final_predictions = ensemble.predict(test, method='mean')
submission = pd.DataFrame({'Id': raw_test['Id'], 'SalePrice': final_predictions})

submission.head()
sample_submission.head()
submission.to_csv('submission3.csv', index=False)