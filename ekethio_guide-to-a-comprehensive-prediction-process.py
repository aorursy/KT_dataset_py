



import numpy as np 

import os

import pandas as pd 

import xgboost as xgb

from xgboost import XGBRegressor

from scipy.stats import boxcox

from sklearn import ensemble, linear_model

from sklearn.preprocessing import LabelEncoder

from sklearn.kernel_ridge import KernelRidge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import SGDRegressor, LinearRegression, BayesianRidge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score

from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_squared_error





train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

ids = test.Id

label = np.log1p(train.pop("SalePrice"))





cols_keep  = train.isnull().sum()[train.isnull().sum() < 0.7 * len(train)].index

train = train[cols_keep]

test = test[cols_keep]

label.drop(train[train["GrLivArea"] > 4000].index, inplace=True)

train.drop(train[train["GrLivArea"] > 4000].index, inplace=True)

ntrain = len(train)



print(train.shape, label.shape)

data = pd.concat([train, test]).reset_index(drop = True)

data.drop(['Id'], axis =1, inplace = True)



# Fill missing values



data['LotFrontage'] = data.LotFrontage.fillna(train['LotFrontage'].median())

data['MasVnrArea'] = data.MasVnrArea.fillna(train.MasVnrArea.median())

data['GarageYrBlt'] = data.GarageYrBlt.fillna(train.GarageYrBlt.median())

data.fillna(data.mode().iloc[0], inplace = True)



print (data.dtypes.value_counts())

num = [col for col in data.columns if data[col].dtype == 'float64' and data[col].dtype ]

cat = [col for col in data.columns if data[col].dtype == 'object']

skew_feats = train[num].skew()[(data[num].skew() > 0.75) | (train[num].skew() < -0.75)].index

for feat in skew_feats:

    data[feat],_ = boxcox(data[feat] + 1)

#train[cat] = train[cat].apply(lambda x: le.fit_transform(x))

data = pd.get_dummies(data)



print (ntrain)

test = data.iloc[ntrain:]

train = data.iloc[:ntrain]

cols = train.columns



xgr = XGBRegressor(max_depth = 4, n_estimators = 200, learning_rate = 0.1, subsample = 0.75, gamma = 0.000001, colsample_bytree = 0.6)

xgr.fit(train, label)

feature_importance = pd.Series({ col: imp for col, imp in zip(cols, xgr.feature_importances_)}) 

print(feature_importance.sort_values(ascending = False).iloc[:20])

features = feature_importance[feature_importance != 0].index

nfold = 10

train = train[features]

test = test[features]



ntrain, ntest = len(train), len(test)

def oof_predictions(train, test, label, models):

    train_scores = np.zeros((ntrain, len(models)))

    test_scores = np.zeros((ntest, len(models)))

    for x in range(len(models)):

        kf = KFold(n_splits = nfold)

        kf = kf.split(train)

        clf = models[x]

        test_fold_scores = np.zeros((ntest,nfold))

        for i, (train_index, test_index) in enumerate(kf):

            train_fold = train[train_index]

            test_fold = train[test_index]

            label_fold = label[train_index]

            clf.fit(train_fold, label_fold)

            train_scores[test_index, x] = clf.predict(test_fold)

            test_fold_scores[:, i] = clf.predict(test)

        test_scores[:, x] = test_fold_scores.mean(axis = 1)

        print(np.sqrt(mean_squared_error(train_scores[:, x], label)))

   

    return train_scores, test_scores



xgr2 = XGBRegressor(max_depth = 3, n_estimators = 200, learning_rate = 0.12, subsample = 0.75, gamma = 0.0001, colsample_bytree = 0.65)

gr = GradientBoostingRegressor(subsample = 0.68, max_depth = 3, learning_rate = 0.11, n_estimators = 200, max_features = 20, min_samples_split = 3)

br = BayesianRidge(alpha_1 = 1e-11, alpha_2 = 0.28, lambda_1 = 1e-11, lambda_2 = 0.0001)

la = linear_model.Lasso(alpha = 0.00015, max_iter = 30000)

pred1, pred2, pred3, pred4 = xgr2.fit(train.values, label).predict(test.values),gr.fit(train.values, label).predict(test.values),br.fit(train.values, label).predict(test.values),la.fit(train.values, label).predict(test.values)

params = { 'gamma': [1e-7, 1e-3, 1e-5, 1e-4]}

'''

gr = GridSearchCV(xgr2, params, cv = 10, scoring = 'neg_mean_squared_error')

gr.fit(train.values, label.values)

print(gr.best_params_)

'''



models = [ xgr2, gr, br]



'''

oof_train, oof_test = oof_predictions(train.values, test.values, label.values, models)

rf = RandomForestRegressor(max_depth = 3, n_estimators = 200)

rf.fit(oof_train, label)

'''

#preds = rf.predict(oof_test)

#oof_test = pd.DataFrame(oof_test, columns = ['XGB', "GradientBoosting", 'BayesianRidge'])

#print(oof_test.corr())

preds = pred3 

#preds = oof_test.mean(axis = 1)

preds = np.exp(preds) - 1





subm = pd.DataFrame()

subm['Id'] = ids

subm['SalePrice'] = preds

print (os.getcwd())



subm.to_csv('ensemble_submission.csv', index = False)
