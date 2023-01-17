import pandas as pd

import numpy as np



from scipy.stats import norm

from scipy import stats

from scipy.stats import skew

from scipy.stats.stats import pearsonr



from sklearn.preprocessing import StandardScaler



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



# Load Data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



#skewness and kurtosis

print("Skewness: %f" % train_df['SalePrice'].skew())

print("Kurtosis: %f" % train_df['SalePrice'].kurt())



#Save the 'Id' column

train_ID = train_df['Id']

test_ID = test_df['Id']

#Now drop the 'Id' colum since it's unnecessary for  the prediction process.

train_df.drop("Id", axis = 1, inplace = True)

test_df.drop("Id", axis = 1, inplace = True)



# preview the data

train_df.head()
quantitative = train_df.dtypes[train_df.dtypes != "object"].index

qualitative = train_df.dtypes[train_df.dtypes == "object"].index



# # pairplot

# def pairplot(x, y, **kwargs):

#     ax = plt.gca()

#     ts = pd.DataFrame({'time': x, 'val': y})

#     ts = ts.groupby('time').mean()

#     ts.plot(ax=ax)

#     plt.xticks(rotation=90)    

# f = pd.melt(train_df, id_vars=['SalePrice'], value_vars=quantitative)

# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

# g = g.map(pairplot, "value", "SalePrice")



# # categorical features

# def boxplot(x, y, **kwargs):

#     sns.boxplot(x=x, y=y)

#     x=plt.xticks(rotation=90)

# f = pd.melt(train_df, id_vars=['SalePrice'], value_vars=qualitative)

# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

# g = g.map(boxplot, "value", "SalePrice")



# # normal distribution

# f = pd.melt(train_df, value_vars=quantitative)

# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)

# g = g.map(sns.distplot, "value")



# normal probability plot

sns.distplot(train_df['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_df['SalePrice'], plot=plt)
#correlation matrix

corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);



#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)



#pairplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_df[cols], size = 2.5)
#  Expensive houses have pools, better overall qual and condition, open porch and increased importance of MasVnrArea.

features = quantitative



standard = train_df[train_df['SalePrice'] < 200000]

pricey = train_df[train_df['SalePrice'] >= 200000]



diff = pd.DataFrame()

diff['feature'] = features

diff['difference'] = [(pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean())/(standard[f].fillna(0.).mean())

                      for f in features]



sns.barplot(data=diff, x='feature', y='difference')

x=plt.xticks(rotation=90)
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(train_df['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)



#bivariate analysis saleprice/grlivarea: scatterplot

var = 'GrLivArea'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));



# Remove outliers

train_df.drop(train_df[(train_df['OverallQual']<5) & (train_df['SalePrice']>200000)].index, inplace=True)

train_df.drop(train_df[(train_df['GrLivArea']>4500) & (train_df['SalePrice']<300000)].index, inplace=True)

train_df.reset_index(drop=True, inplace=True)



ntrain = train_df.shape[0]

ntest = test_df.shape[0]

# combine these datasets to run certain operations on both datasets together

full_data = pd.concat((train_df, test_df)).reset_index(drop=True)

full_data.drop(['SalePrice'], axis=1, inplace=True)
# missing data

total = full_data.isnull().sum().sort_values(ascending=False)

percent = (full_data.isnull().sum()/full_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)



#dealing with missing data

#train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)



full_data["LotFrontage"] = full_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

full_data.fillna(full_data.mean(), inplace=True)

full_data.fillna(full_data.mode().iloc[0], inplace=True)



full_data.isnull().sum().max() #just checking that there's no more data missing...
# from sklearn.preprocessing import LabelEncoder



# # transform into a categorical variable

# full_data['MSSubClass'] = full_data['MSSubClass'].apply(str)

# full_data['OverallCond'] = full_data['OverallCond'].astype(str)

# full_data['YrSold'] = full_data['YrSold'].astype(str)

# full_data['MoSold'] = full_data['MoSold'].astype(str)



# cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

#         'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

#         'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

#         'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

#         'YrSold', 'MoSold')

# # process columns, apply LabelEncoder to categorical features

# for c in cols:

#     lbl = LabelEncoder() 

#     lbl.fit(list(full_data[c].values)) 

#     full_data[c] = lbl.transform(list(full_data[c].values))

     

# print('Shape of full_data: {}'.format(full_data.shape))
# Adding total sqfootage feature 

full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']

full_data = full_data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1)

print('Shape of full_data: {}'.format(full_data.shape))
from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# normality by applying log transformation

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])



#log transform skewed numeric features

numeric_feats = full_data.dtypes[full_data.dtypes != "object"].index

skewed_feats = full_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skew_index = skewed_feats.index

print(skew_index)



# Normalize skewed features

for i in skew_index:

    full_data[i] = boxcox1p(full_data[i], boxcox_normmax(full_data[i] + 1))
# convert categorical variable into dummy

full_data = pd.get_dummies(full_data)



full_data.head()
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, Lasso, LassoCV, LassoLarsCV, BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb



# data matrices for learning

X_train = full_data[:ntrain]

X_test = full_data[ntrain:]

y = train_df.SalePrice.values



# Validation function

kf = KFold(n_splits=12, random_state=42, shuffle=True)

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]



cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation")

plt.xlabel("alpha")

plt.ylabel("rmse")



print(cv_ridge.min())



model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

print(rmse_cv(model_lasso).mean())



#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")



dtrain = xgb.DMatrix(X_train, label = y)

params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()



# model_xgb = xgb.XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

# model_xgb.fit(X_train, y)



# xgb_preds = np.expm1(model_xgb.predict(X_test))

# lasso_preds = np.expm1(model_lasso.predict(X_test))

# preds = 0.7*lasso_preds + 0.3*xgb_preds
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



# Setup models

lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=7000,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)



xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=42)



# Ridge Regressor

ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))



# Support Vector Regressor

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))



# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)  



# Random Forest Regressor

rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=42)



# Stack up all the models above, optimized using xgboost

stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)



# Get cross validation scores for each model

scores = {}



score = rmse_cv(lightgbm)

print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lgb'] = (score.mean(), score.std())



score = rmse_cv(xgboost)

print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['xgb'] = (score.mean(), score.std())



score = rmse_cv(svr)

print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['svr'] = (score.mean(), score.std())



score = rmse_cv(ridge)

print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['ridge'] = (score.mean(), score.std())



score = rmse_cv(rf)

print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['rf'] = (score.mean(), score.std())



score = rmse_cv(gbr)

print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['gbr'] = (score.mean(), score.std())



# Plot the predictions for each model

sns.set_style("white")

fig = plt.figure(figsize=(24, 12))



ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])

for i, score in enumerate(scores.values()):

    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')



plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)

plt.xlabel('Model', size=20, labelpad=12.5)

plt.tick_params(axis='x', labelsize=13.5)

plt.tick_params(axis='y', labelsize=12.5)



plt.title('Scores of Models', size=20)



plt.show()
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

    

# stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

#                                                  meta_model = lasso)



# score = rmse_cv(stacked_averaged_models)

# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X_train.values), np.array(y))



print('lightgbm')

lgb_model_full_data = lightgbm.fit(X_train, y)



print('xgboost')

xgb_model_full_data = xgboost.fit(X_train, y)



print('Svr')

svr_model_full_data = svr.fit(X_train, y)



print('Ridge')

ridge_model_full_data = ridge.fit(X_train, y)



print('RandomForest')

rf_model_full_data = rf.fit(X_train, y)



print('GradientBoosting')

gbr_model_full_data = gbr.fit(X_train, y)





def blended_predictions(X):

    return ((0.1 * ridge_model_full_data.predict(X)) + \

            (0.2 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.1 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.05 * rf_model_full_data.predict(X)) + \

            (0.35 * stack_gen_model.predict(np.array(X))))



# Get final precitions from the blended model

blended_score = rmsle(y, blended_predictions(X_train))

scores['blended'] = (blended_score, 0)

print('RMSLE score on train data:')

print(blended_score)
predictions = blended_predictions(X_test)



submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = np.floor(np.expm1(predictions))

submission.to_csv('submission.csv',index=False)