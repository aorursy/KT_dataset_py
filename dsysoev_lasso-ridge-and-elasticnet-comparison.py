import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_style('white')
# load data

train = pd.read_csv('../input/train.csv', index_col='Id')

test = pd.read_csv('../input/test.csv', index_col='Id')

train.tail()
# convert int values to str for specific features

for categ in ['MSSubClass', 'OverallQual', 'OverallCond']:

    train[categ] = train[categ].astype(str)

    test[categ] = test[categ].astype(str)
sns.distplot(train['SalePrice'])
# show features with count of NaN values

s = train.isnull().sum() + test.isnull().sum()

s = s.sort_values(ascending=False)

s = s[s > 0]

s
# unique names for features with NaN values

for i in s.index:

    print(i, train[i].unique().tolist()[:10])
for feature in ['GarageYrBlt', 'LotFrontage', 'MasVnrArea', 'BsmtFullBath',

                'BsmtFinSF1', 'GarageArea', 'BsmtFinSF2', 'TotalBsmtSF',

                'BsmtUnfSF', 'BsmtHalfBath', 'GarageCars']:

    train[feature] = train[feature].fillna(train[feature].mean())

    test[feature] = test[feature].fillna(test[feature].mean())



for feature in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',

                'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual',

                'BsmtFinType2','BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1',

                'Electrical', 'MSZoning', 'Functional', 'Utilities', 'KitchenQual',

                'SaleType', 'Exterior1st', 'Exterior2nd'

               ]:

    train[feature] = train[feature].fillna('NaN')

    test[feature] = test[feature].fillna('NaN')

    

for feature in ['MasVnrType']:

    train[feature] = train[feature].fillna('None')

    test[feature] = test[feature].fillna('None')
# check NaN values

s = train.isnull().sum() + test.isnull().sum()

s = s.sort_values(ascending=False)

s = s[s > 0]

s
# select features for category 

d = train.dtypes.groupby(train.dtypes).groups

category = d[np.dtype('O')].tolist() + ['MSSubClass', 'OverallQual', 'OverallCond']

# category
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction import DictVectorizer
# fit DictVectorizer and StandardScaler on whole data

whole_data = pd.concat([train.drop('SalePrice', axis=1).iloc[0:-1], test], axis=0)



vec = DictVectorizer()

vec.fit(whole_data[category].to_dict('records'))



scaler = StandardScaler()

scaler.fit(whole_data.drop(category, axis=1))
train_without_category = train.drop(category + ['SalePrice'], axis=1)

# create DataFrame with category features

X_category = vec.transform(train[category].to_dict('records'))

train_category = pd.DataFrame(X_category.toarray(), columns=vec.feature_names_)

X_scale = scaler.transform(train_without_category)

# create DataFrame with scaled features

train_scale = pd.DataFrame(X_scale, columns=train_without_category.columns)

# create final train DataFrame 

train_final = pd.concat([train_scale, train_category], axis=1)

train_final.tail()
X_category = vec.transform(test[category].to_dict('records'))

# create DataFrame with category features

test_category = pd.DataFrame(X_category.toarray(), columns=vec.feature_names_)

test_without_category = test.drop(category, axis=1)

X_scale = scaler.transform(test_without_category)

# create DataFrame with scaled features

test_scale = pd.DataFrame(X_scale, columns=test_without_category.columns)

# create final test DataFrame 

test_final = pd.concat([test_scale, test_category], axis=1)

test_final.tail()
plt.title('train')

_  = plt.plot(train_final)

plt.show()

plt.title('test')

_  = plt.plot(test_final)
## Log transformation of y

X_train, y_train = train_final.values, np.log(train['SalePrice'].values)

X_test = test_final.values
from sklearn.linear_model import Lasso

from sklearn.linear_model import LassoCV

from sklearn.linear_model import Ridge

from sklearn.linear_model import RidgeCV

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import ElasticNetCV
# dict with optimal models

models = {}

# find optimal value of alpha 

n_trials = 100

alpha_list = 10 ** np.linspace(-5, 5, n_trials)

# number of folds for cross validation

cv = 5

# find optimal value of l1 (for ElasticNet)

l1_list = 10 ** np.linspace(-2, 0, 50)

max_iter = 5000
# find optimal Lasso model

clf = LassoCV(alphas=alpha_list, cv=cv, n_jobs=-1, random_state=1, max_iter=max_iter)

clf.fit(X_train, y_train)

models['Lasso'] = Lasso(alpha=clf.alpha_, max_iter=max_iter)
# find optimal Ridge model

clf = RidgeCV(alphas=alpha_list, cv=cv)

clf.fit(X_train, y_train)

models['Ridge'] = Ridge(alpha=clf.alpha_)
# find optimal ElasticNet model

clf = ElasticNetCV(alphas=alpha_list, l1_ratio=l1_list,

                   cv=cv, random_state=1, n_jobs=-1, max_iter=max_iter)

clf.fit(X_train, y_train)

models['ElasticNet'] = ElasticNet(alpha=clf.alpha_, l1_ratio=clf.l1_ratio_, max_iter=max_iter)
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
# final cross validation between models

kf = KFold(cv, shuffle=True, random_state=1)



score = {}

for name in models:

    # save score for each model

    if name not in score:

        score[name] = []

    clf = models[name]

    for i_train, i_test in kf.split(X_train):

        clf.fit(X_train[i_train], y_train[i_train])

        y_pred = clf.predict(X_train[i_test])

        RMSE = np.sqrt(mean_squared_error(y_train[i_test], y_pred))

        score[name].append(RMSE)

# results for all models

results = pd.DataFrame(score)
for key in results:

    _ = plt.plot(results[key], label=key)

plt.legend()

pd.concat([results.mean(), results.std()], axis=1, keys=['mean', 'std'])
# create drop_features list with features with l1 coef_ == 0

s = pd.Series(dict(zip(train_final.columns, models['ElasticNet'].coef_))).abs().sort_values(ascending=True)

drop_features = s[s == 0].index.tolist()

len(drop_features)
# remove some features and create new train and test

X_train = train_final.drop(drop_features, axis=1).values

X_test = test_final.drop(drop_features, axis=1)

# create final model for submission

clf = models['ElasticNet'].fit(X_train, y_train)

y_pred = clf.predict(X_train)

# show score

np.sqrt(mean_squared_error(y_train, y_pred))
# create submission with best model

y_pred = clf.predict(X_test)

pred_df = pd.DataFrame(np.exp(y_pred), index=test.index, columns=["SalePrice"])

pred_df.to_csv('submission.csv', header=True, index_label='Id')
# comments are welcome)