import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.svm import SVR
# Display all columns of a dataframe

pd.pandas.set_option("display.max_columns", None)
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train.head()
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

test.head()
print("Train Dataset: ", train.shape)

print("Test Dataset: ", test.shape)
train.describe()
train.info()
# Sales Price Distribution

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(6,5))

sns.distplot(train['SalePrice'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="Sales Price")

ax.set(title="Home Sales Price Distribution")

sns.despine(trim=True, left=True)

plt.show()
# Skew and kurt

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
plt.figure(figsize=(20,10))

sns.heatmap(data=train.corr(), cmap="Blues", square=True)

plt.show()
# Get features with missing values

features_with_na = [feature for feature in train.columns if train[feature].isnull().sum() > 0]
# Print missing features and its percentage in train dataset

for feature in features_with_na:

    print(feature, np.round(train[feature].isnull().mean(), 4), "% missing values")
for feature in features_with_na:

    data = train.copy()

    #Create a variable that indicates 1 if the values is missing and 0 otherwise.

    data[feature] = np.where(data[feature].isnull(), 1, 0)

    

    # Plot bar graph of median SalesPrice for values missing or present in train dataset

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel("SalePrice")

    plt.title(feature)

    plt.show()
numerical_features = [feature for feature in train.columns if train[feature].dtype != 'O']

print("Number of numerical features: ", len(numerical_features))

train[numerical_features].head()
temporal_features = [feature for feature in numerical_features if 'Year' in feature or 'Yr' in feature]

print("Number of temporal features: ", len(temporal_features))

train[temporal_features].head()
for feature in temporal_features:

    data = train.copy()

    

    data.groupby(feature)['SalePrice'].median().plot()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
for feature in temporal_features:

    data = train.copy()

    

    if feature != 'YrSold':

        data[feature] = data['YrSold'] - data[feature]

        plt.scatter(data[feature], data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.title(feature)

        plt.show()
discrete_features = [feature for feature in numerical_features if len(train[feature].unique()) <=25 

                     and feature not in temporal_features + ['Id']]

print("Length of discrete features: ", len(discrete_features))

train[discrete_features].head()
for feature in discrete_features:

    data = train.copy()

    

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
continuous_features = [feature for feature in numerical_features if feature not in discrete_features + temporal_features + ['Id']]

print("Length of continuous features: ", len(continuous_features))

train[continuous_features].head()
for feature in continuous_features:

    data = train.copy()

    

    data[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel("Count")

    plt.title(feature)

    plt.show()
for feature in continuous_features:

    data = train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature] = np.log(data[feature])

        plt.scatter(data[feature], data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('Sale Price')

        plt.title(feature)

        plt.show()
for feature in continuous_features:

    data = train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature] = np.log(data[feature])

        data.boxplot(column=feature)

        plt.title(feature)

        plt.show()
categorial_features = [feature for feature in train.columns if train[feature].dtypes == 'O']

print(categorial_features)
train[categorial_features].head()
for feature in categorial_features:

    print("Feature {} has {} unique values".format(feature, len(train[feature].unique())))
for feature in categorial_features:

    data = train.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('Sale Price')

    plt.title(feature)

    plt.show()
train[categorial_features].head()
categorial_with_nan = [feature for feature in categorial_features if train[feature].isnull().sum() > 0]

print(categorial_with_nan)

for feature in categorial_with_nan:

    print("Feature {}, has {}% missing values in train dataset", (feature, np.round(train[feature].isnull().mean(), 4)))
for feature in categorial_with_nan:

    train[feature].fillna('Missing', inplace=True)
categorial_with_nan = [feature for feature in categorial_features if test[feature].isnull().sum() > 0]

print(categorial_with_nan)

for feature in categorial_with_nan:

    print("Feature {}, has {}% missing values in test dataset", (feature, np.round(test[feature].isnull().mean(), 4)))
for feature in categorial_with_nan:

    test[feature].fillna('Missing', inplace=True)
train[categorial_features].head()
test[categorial_features].head()
print("Train Dataset Categorial Features:",train[categorial_features].shape)

print("Test Dataset Categorial Features:",test[categorial_features].shape)
print(numerical_features)
numerical_with_nan = [feature for feature in numerical_features if train[feature].isnull().sum() > 0]

print(numerical_with_nan)

for feature in numerical_with_nan:

    print("Feature {} has {}% missing values in train datset", (feature,np.round(train[feature].isnull().mean(), 4)))
for feature in numerical_with_nan:

    train[feature].fillna(train[feature].median(), inplace=True)
train.head()
numerical_with_nan = [feature for feature in numerical_features if feature not in ['SalePrice'] and test[feature].isnull().sum() > 0]

print(numerical_with_nan)

for feature in numerical_with_nan:

    print("Feature {} has {}% missing values in test datset", (feature,np.round(test[feature].isnull().mean(), 4)))
for feature in numerical_with_nan:

    test[feature].fillna(test[feature].median(), inplace=True)
test.head()
print("Train Dataset", train.shape)

print("Test Dataset", test.shape)
train[temporal_features].head()
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:

    train[feature] = train['YrSold'] - train[feature]

    test[feature] = test['YrSold'] - test[feature]
train[temporal_features].head()
test[temporal_features].head()
train.head()
num_non_zero_skewed_features_train_set = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

train[num_non_zero_skewed_features_train_set].head()
for feature in num_non_zero_skewed_features_train_set:

    train[feature] = np.log(train[feature])
num_non_zero_skewed_features_test_set = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

test[num_non_zero_skewed_features_test_set].head()
for feature in num_non_zero_skewed_features_test_set:

    test[feature] = np.log(test[feature])
train[num_non_zero_skewed_features_train_set].head()
test[num_non_zero_skewed_features_test_set].head()
train[categorial_features].head()
print(train.shape)

print(test.shape)
print(len(categorial_features))

print(len(numerical_features))
remaining_features = [feature for feature in train.columns if feature not in categorial_features + numerical_features]

print(remaining_features)
train1 = train.copy()

test1 = test.copy()
data = pd.concat([train1,test1], axis=0)

train_rows = train1.shape[0]



for feature in categorial_features:

    dummy = pd.get_dummies(data[feature])

    for col_name in dummy.columns:

        dummy.rename(columns={col_name: feature+"_"+col_name}, inplace=True)

    data = pd.concat([data, dummy], axis = 1)

    data.drop([feature], axis = 1, inplace=True)



train1 = data.iloc[:train_rows, :]

test1 = data.iloc[train_rows:, :] 
train1.head()
test1.head()
print("Train",train1.shape)

print("Test",test1.shape)
from sklearn.preprocessing import MinMaxScaler, RobustScaler



scaling_features = [feature for feature in train1.columns if feature not in ['Id', 'SalePrice']]

scaling_features
print(len(scaling_features))
train1[scaling_features].head()
scaler = RobustScaler()

scaler.fit(train1[scaling_features])
X_train = scaler.transform(train1[scaling_features])

X_test = scaler.transform(test1[scaling_features])
print("Train", X_train.shape)

print("Test", X_test.shape)
y_train = train1['SalePrice']
X = pd.concat([train1[['Id','SalePrice']].reset_index(drop=True), pd.DataFrame(X_train, columns = scaling_features)], axis =1)

print(X.shape)

X.head()
n_folds = 12



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



def rmsle(y_train, y_pred):

    return np.sqrt(mean_squared_error(y_train, y_pred))
lasso = Lasso(alpha =0.0005, random_state=0)

elasticNet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=0)

kernelRidge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

svr = SVR(C= 20, epsilon= 0.008, gamma=0.0003)

gradientBoosting = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =0)

xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =0, nthread = -1)

lgbm = LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11, random_state=0)

randomForest = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=0)
scores ={}
score = rmsle_cv(lasso)

print("Lasso:: Mean:",score.mean(), " Std:", score.std())

scores['lasso'] = (score.mean(), score.std())

lasso_model = lasso.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_train)

rmsle(y_train,y_pred_lasso)
score = rmsle_cv(elasticNet)

print("ElasticNet:: Mean:",score.mean(), " Std:", score.std())

scores['elasticNet'] = (score.mean(), score.std())

elasticNet_model = elasticNet.fit(X_train, y_train)

y_pred_elasticNet = elasticNet_model.predict(X_train)

rmsle(y_train,y_pred_elasticNet)
score = rmsle_cv(kernelRidge)

print("KernelRidge:: Mean:",score.mean(), " Std:", score.std())

scores['kernelRidge'] = (score.mean(), score.std())

kernelRidge_model = kernelRidge.fit(X_train, y_train)

y_pred_kernelRidge = kernelRidge_model.predict(X_train)

rmsle(y_train,y_pred_kernelRidge)
score = rmsle_cv(svr)

print("SVR:: Mean:",score.mean(), " Std:", score.std())

scores['svr'] = (score.mean(), score.std())

svr_model = svr.fit(X_train, y_train)

y_pred_svr = svr_model.predict(X_train)

rmsle(y_train,y_pred_svr)
score = rmsle_cv(gradientBoosting)

print("GradientBoostingRegressor:: Mean:",score.mean(), " Std:", score.std())

scores['gradientBoosting'] = (score.mean(), score.std())

gradientBoosting_model = gradientBoosting.fit(X_train, y_train)

y_pred_gradientBoosting = gradientBoosting_model.predict(X_train)

rmsle(y_train,y_pred_gradientBoosting)
score = rmsle_cv(xgb)

print("XGBRegressor:: Mean:",score.mean(), " Std:", score.std())

scores['xgb'] = (score.mean(), score.std())

xgb_model = xgb.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_train)

rmsle(y_train,y_pred_xgb)
score = rmsle_cv(lgbm)

print("LGBMRegressor:: Mean:",score.mean(), " Std:", score.std())

scores['lgbm'] = (score.mean(), score.std())

lgbm_model = lgbm.fit(X_train, y_train)

y_pred_lgbm = lgbm_model.predict(X_train)

rmsle(y_train,y_pred_lgbm)
score = rmsle_cv(randomForest)

print("RandomForestRegressor:: Mean:",score.mean(), " Std:", score.std())

scores['randomForest'] = (score.mean(), score.std())

randomForest_model = randomForest.fit(X_train, y_train)

y_pred_randomForest = randomForest_model.predict(X_train)

rmsle(y_train,y_pred_randomForest)
def ensemble_models(X):

    return ((0.1 * lasso_model.predict(X)) +

            (0.1 * elasticNet_model.predict(X)) +

            (0.1 * kernelRidge_model.predict(X)) +

            (0.1 * svr_model.predict(X)) +

            (0.2 * gradientBoosting_model.predict(X)) + 

            (0.1 * xgb_model.predict(X)) +

            (0.2 * lgbm_model.predict(X)) +

            (0.1 * randomForest_model.predict(X)))
averaged_score = rmsle(y_train, ensemble_models(X_train))

scores['averaged'] = (averaged_score, 0)

print('RMSLE averaged score on train data:', averaged_score)
def stack_models(X):

    return ((0.7 * ensemble_models(X)) +

            (0.15 * lasso_model.predict(X)) +

#             (0.1 * elasticNet_model.predict(X)) +

#             (0.1 * gradientBoosting_model.predict(X)) + 

            (0.15 * xgb_model.predict(X))

#             (0.15 * lgbm_model.predict(X))

           )
stacked_score = rmsle(y_train, stack_models(X_train))

scores['stacked'] = (stacked_score, 0)

print('RMSLE stacked score on train data:', stacked_score)
sns.set_style("white")

fig = plt.figure(figsize=(20, 10))



ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])

for i, score in enumerate(scores.values()):

    ax.text(i, score[0] + 0.002, '{:.4f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')



plt.ylabel('Score', size=20, labelpad=12.5)

plt.xlabel('Regression Model', size=20, labelpad=12.5)

plt.tick_params(axis='x', labelsize=13.5)

plt.tick_params(axis='y', labelsize=12.5)

plt.title('Regression Model Scores', size=20)

plt.show()
test_predict = np.exp(stack_models(X_test))

print(test_predict[:5])
sub = pd.DataFrame()

sub['Id'] = test['Id']

sub['SalePrice'] = test_predict

sub.to_csv('submission.csv',index=False)
sub1 = pd.read_csv('submission.csv')

sub1.head()