import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



# sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col=0)

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col=0)
train.shape
train.dtypes.value_counts()
grouped = train.columns.to_series().groupby(train.dtypes).groups

print(grouped)
train.head()
for col in train.columns:

    try:

        empty = train[col].isna().value_counts()[1]

    except KeyError as e:

        empty = 0

    ratio = empty / train.shape[0]

    if ratio > 0.25:

        print("{}: {:0.2f}".format(col, ratio))
train2 = train.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])



train.head()
print(train2.shape)

print(train2.dtypes.value_counts())



grouped = train2.columns.to_series().groupby(train.dtypes).groups
objs = list(grouped.values())[2]

dense = []



for col in objs:

    categs = train[col].nunique()

    if categs > 2:

        dense.append(col)



print(train[dense].shape)

train[dense].describe()
train[objs].nunique().median()
very_dense = []



for col in dense:

    categs = train[col].nunique()

    if categs > 5:

        very_dense.append(col)



print(train[very_dense].shape)

train[very_dense].describe()
from scipy.stats import pearsonr



target = train['SalePrice']



Pearson = {}



for categ in very_dense:

    OHdf = pd.get_dummies(train[categ])

    pearson = np.empty(OHdf.shape[1])

    pval = np.empty(OHdf.shape[1])

    for i, col in enumerate(OHdf.columns):

        pearson[i], pval[i] = pearsonr(OHdf[col], target)

    Pearson[categ] = {'r': np.mean(pearson), 'p': np.mean(pval)}

    print("{}: r: {:0.2f}, p: {:0.2f}".format(categ, Pearson[categ]['r'], Pearson[categ]['p']))
dense_relevant = []



for col, pearson in Pearson.items():

    if pearson['p'] < 0.05:

        dense_relevant.append(col)



print(dense_relevant)
train['BsmtFinType1'].unique()
ind = train2.index



DenseDf = pd.DataFrame(index=ind)



for col in [x for x in dense if x not in very_dense]:

    OHdf = pd.get_dummies(train[col], prefix=col)

    DenseDf = pd.concat([DenseDf, OHdf], axis=1)



DenseDf = pd.concat([DenseDf, pd.get_dummies(train['BsmtFinType1'], prefix='Bsmt')], axis=1)



DenseDf.head()
binaries = [x for x in objs if x not in dense]



print(binaries)
train['Street'].unique() # Pave va a ser 1
train['Utilities'].unique() # AllPub va a ser 1
train['CentralAir'].unique() # Y va a ser 1
BNdf = train[binaries]

BNen = pd.DataFrame(index=ind)



BNyes = {'Street':'Pave','Alley':'Pave','Utilities':'AllPub','CentralAir':'Y'}



for col in BNdf.columns:

    yes = pd.Series( (train[col] == BNyes[col]).astype(int), index=ind, name=col )

    BNen = pd.concat([BNen, yes], axis=1)



BNen.head()
CategDf = pd.concat([BNen, DenseDf], axis=1)

CategDf.head()
good = [x for x in train2.columns if x not in objs]

good.remove('SalePrice')



# Engineered DataFrames

dfX = pd.concat([train2[good], CategDf], axis=1)

dfY = train2['SalePrice'].astype(float)



dfX.shape
fig, axes = plt.subplots(6, 6, figsize=(20, 20))



for i, col in enumerate(dfX[good]):

    sns.boxplot(x=col, data=dfX, ax=axes.flat[i])
# importar modelos necesarios de sklearn

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error, mean_absolute_error
X_train, X_validate, y_train, y_validate = train_test_split(dfX, dfY, random_state=42)



model = Pipeline([("imputer", SimpleImputer(strategy='mean')),

                  ("scaler", RobustScaler()),

                  ("regression", Lasso())])

model.fit(X_train, y_train)

print("Training set score: {:.2f}".format(model.score(X_train, y_train)))

print("Training set mse: {:.2f}".format(mean_squared_error(y_train, model.predict(X_train))))

print("Training set mae: {:.2f}\n".format(mean_absolute_error(y_train, model.predict(X_train))))



print("Validation set score: {:.2f}".format(model.score(X_validate, y_validate)))

print("Validation set mse: {:.2f}".format(mean_squared_error(y_validate, model.predict(X_validate))))

print("Validation set mae: {:.2f}".format(mean_absolute_error(y_validate, model.predict(X_validate))))
from sklearn.ensemble import RandomForestRegressor



forestM = Pipeline([("imputer", SimpleImputer(strategy="mean")),

                   ("regressor", RandomForestRegressor(random_state=42))])



trees = np.arange(1, 100)

param_grid = {'regressor__n_estimators':trees}



forest = GridSearchCV(forestM, param_grid=param_grid, cv=5, n_jobs=-1)

forest.fit(X_train, y_train)



print("Best params: {}\n".format(forest.best_params_))



print("Training set score: {:.2f}".format(forest.score(X_train, y_train)))

print("Training set mse: {:.2f}".format(mean_squared_error(y_train, forest.predict(X_train))))

print("Training set mae: {:.2f}\n".format(mean_absolute_error(y_train, forest.predict(X_train))))



print("Validation set score: {:.2f}".format(forest.score(X_validate, y_validate)))

print("Validation set mse: {:.2f}".format(mean_squared_error(y_validate, forest.predict(X_validate))))

print("Validation set mae: {:.2f}".format(mean_absolute_error(y_validate, forest.predict(X_validate))))
X_test = test.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])



t_ind = X_test.index



t_numeric = [x for x in X_test.columns if x not in objs]



t_BNdf = test[binaries]

t_BNen = pd.DataFrame(index=t_ind)



for col in t_BNdf.columns:

    yes = pd.Series( (test[col] == BNyes[col]).astype(int), index=t_ind, name=col )

    t_BNen = pd.concat([t_BNen, yes], axis=1)



t_BNen.head()
t_DenseDf = pd.DataFrame(index=t_ind)



for col in [x for x in dense if x not in very_dense]:

    OHdf = pd.get_dummies(X_test[col], prefix=col)

    t_DenseDf = pd.concat([t_DenseDf, OHdf], axis=1)



t_DenseDf = pd.concat([t_DenseDf, pd.get_dummies(X_test['BsmtFinType1'], prefix='Bsmt')], axis=1)



print("Train dense shape: {}".format(DenseDf.shape))

print("Test dense shape: {}".format(t_DenseDf.shape))



t_DenseDf.head()
print([x for x in DenseDf.columns if x not in t_DenseDf.columns])
for col in ['Electrical_Mix', 'GarageQual_Ex']:

    t_DenseDf[col] = np.zeros(shape=t_ind.shape, dtype=int)

    

print("Train dense shape: {}".format(DenseDf.shape))

print("Test dense shape: {}".format(t_DenseDf.shape))
X_test = pd.concat([X_test[t_numeric], t_BNen, t_DenseDf], axis=1)



print("Train shape: {}".format(dfX.shape))

print("Test shape: {}".format(X_test.shape))
y_pred = forest.predict(X_test)



y_pred.shape
submission = pd.Series(data=y_pred, index=t_ind, name='SalePrice')



submission.head()