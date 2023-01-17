import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from datetime import datetime



from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print("Train set size:", train.shape)

print("Test set size:", test.shape)
train_ID = train['Id']

test_ID = test['Id']



# Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
corr = train.corr()

plt.subplots(figsize=(13,10))

sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
train.get_dtype_counts()
# Heatmap between top 10 correlated variables with SalePrice

corrMatrix=train[["SalePrice","OverallQual","GrLivArea","GarageCars",

                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()



sns.set(font_scale=1.10)

plt.figure(figsize=(10, 10))



sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='viridis',linecolor="white")

plt.title('Correlation between features');
sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the new distribution 

sns.distplot(train['SalePrice'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)

plt.show()
print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
sns.factorplot("Fireplaces","SalePrice",data=train,hue="FireplaceQu");
labels = train["MSZoning"].unique()

sizes = train["MSZoning"].value_counts().values

explode=[0.1,0,0,0,0]

parcent = 100.*sizes/sizes.sum()

labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]



colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral','blue']

patches, texts= plt.pie(sizes, colors=colors,explode=explode,

                        shadow=True,startangle=90)

plt.legend(patches, labels, loc="best")



plt.title("Zoning Classification")

plt.show()
train1 = train

train1['SalePriceSF'] = train['SalePrice']/train['GrLivArea']

plt.hist(train['SalePriceSF'], bins=15,color="blue")

plt.title("Sale Price per Square Foot")

plt.ylabel('Number of Sales')

plt.xlabel('Price per square feet');
train1['ConstructionAge'] = train['YrSold'] - train['YearBuilt']

plt.scatter(train1['ConstructionAge'], train['SalePriceSF'])

plt.ylabel('Price per square foot (in dollars)')

plt.xlabel("Construction Age of house");
sns.stripplot(x="HeatingQC", y="SalePrice",data=train,hue='CentralAir',jitter=True,split=True)

plt.title("Sale Price vs Heating Quality");
sns.factorplot("KitchenAbvGr","SalePrice",data=train,hue="KitchenQual")

plt.title("Sale Price vs Kitchen");
plt.style.use('seaborn')

plt.scatter(train.GrLivArea, train.SalePrice)
# Deleting outliers

train = train[train.GrLivArea < 4500]

train.reset_index(drop=True, inplace=True)
y = train['SalePrice']

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=stats.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=stats.lognorm)
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train.SalePrice.reset_index(drop=True)

train_features = train.drop(['SalePrice'], axis=1)

test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)

print(features.shape)
# Some of the non-numeric predictors are stored as numbers; we convert them into strings 

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)



features['Functional'] = features['Functional'].fillna('Typ')

features['Electrical'] = features['Electrical'].fillna("SBrkr")

features['KitchenQual'] = features['KitchenQual'].fillna("TA")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

features["PoolQC"] = features["PoolQC"].fillna("None")
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')

features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)



features.update(features[objects].fillna('None'))
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# Filling in the rest of the NA's



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)



skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



# simplified features

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
print(features.shape)

final_features = pd.get_dummies(features).reset_index(drop=True)

print(final_features.shape)



X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(X):, :]



print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)
outliers = [30, 88, 462, 631, 1322]

X = X.drop(X.index[outliers])

y = y.drop(y.index[outliers])



overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.94:

        overfit.append(i)



overfit = list(overfit)

overfit.append('MSZoning_C (all)')



X = X.drop(overfit, axis=1).copy()

X_sub = X_sub.drop(overfit, axis=1).copy()



print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)
kfolds = KFold(n_splits=8, shuffle=True, random_state=42)
# rmsle

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))





# build our model scoring function

def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y,

                                    scoring="neg_mean_squared_error",

                                    cv=kfolds))

    return (rmse)
# setup models    

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



# Ridge Regressor

ridge = make_pipeline(RobustScaler(),

                      RidgeCV(alphas=alphas_alt, cv=kfolds))



# Lasso Regressor

lasso = make_pipeline(RobustScaler(),

                      LassoCV(max_iter=1e7, alphas=alphas2,

                              random_state=42, cv=kfolds))



# Elasticnet Regressor

elasticnet = make_pipeline(RobustScaler(),

                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,

                                        cv=kfolds, l1_ratio=e_l1ratio))

                                        

# Support Vector Regressor

svr = make_pipeline(RobustScaler(),

                      SVR(C= 20, epsilon= 0.008, gamma=0.0003,))



# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =42)

                                   

# Light Gradient Boosting Regressor

lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       #min_data_in_leaf=2,

                                       #min_sum_hessian_in_leaf=11

                                       )

                                       

# XGBoost Regressor

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)



# Stack up all the models above, optimized using xgboost

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,

                                            gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
print('TEST score on CV')



scores = {}



score = cv_rmse(ridge)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()),)

scores['ridge'] = (score.mean(), score.std())



score = cv_rmse(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), )

scores['lasso'] = (score.mean(), score.std())



score = cv_rmse(elasticnet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()),)

scores['elasticnet'] = (score.mean(), score.std())



score = cv_rmse(svr)

print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()),)

scores['svr'] = (score.mean(), score.std())



score = cv_rmse(lightgbm)

print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), )

scores['lgbm'] = (score.mean(), score.std())



score = cv_rmse(gbr)

print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()),)

scores['gbr'] = (score.mean(), score.std())



score = cv_rmse(xgboost)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), )

scores['xgboost'] = (score.mean(), score.std())
print('START Fit')

print(datetime.now(), 'StackingCVRegressor')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print(datetime.now(), 'elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)

print(datetime.now(), 'lasso')

lasso_model_full_data = lasso.fit(X, y)

print(datetime.now(), 'ridge')

ridge_model_full_data = ridge.fit(X, y)

print(datetime.now(), 'svr')

svr_model_full_data = svr.fit(X, y)

print(datetime.now(), 'GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)

print(datetime.now(), 'xgboost')

xgb_model_full_data = xgboost.fit(X, y)

print(datetime.now(), 'lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
def blend_models_predict(X):

    return ((0.1 * elastic_model_full_data.predict(X)) + \

            (0.1 * lasso_model_full_data.predict(X)) + \

            (0.1 * ridge_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.25 * stack_gen_model.predict(np.array(X))))

            



# Get final precitions from the blended model

blended_score = rmsle(y, blend_models_predict(X))

scores['blended'] = (blended_score, 0)

print('RMSLE score on train data:')

print(blended_score)
print('Predict submission', datetime.now(),)

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))



# this kernel gives lower score

# let's up it by mixing with the top kernels



print('Blend with Top Kernals submissions', datetime.now(),)

sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')

sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')

sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')



submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 

                                (0.25 * sub_1.iloc[:,1]) + 

                                (0.25 * sub_2.iloc[:,1]) + 

                                (0.25 * sub_3.iloc[:,1]))
# Brutal approach to deal with predictions close to outer range 

q1 = submission['SalePrice'].quantile(0.0045)

q2 = submission['SalePrice'].quantile(0.99)



submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)



submission.to_csv("submission.csv", index=False)

print('Save submission', datetime.now(),)
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



plt.title('Scores of Various Models', size=20)



plt.show()