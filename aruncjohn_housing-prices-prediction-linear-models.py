import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

from numpy import unique, log1p



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Models

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA



warnings.filterwarnings('ignore')

%matplotlib inline
train_file = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'

train_df = pd.read_csv(train_file)

test_file = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

test_df = pd.read_csv(test_file)
train_df.head(10)

train_df['SalePrice'].describe()
train_df.OverallQual.values[:25]
train_df.SalePrice.plot.hist(bins=50)
#histogram

sns.distplot(train_df['SalePrice'])
#skewness and kurtosis

print("Skewness: %f" % train_df['SalePrice'].skew())

print("Kurtosis: %f" % train_df['SalePrice'].kurt())
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_df[cols], size = 2.5)

plt.show()
train_df.columns
data = pd.concat([train_df.SalePrice, train_df.GrLivArea], axis = 1)

data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
ind_list = train_df.index[train_df['GrLivArea'].values >= 4500].tolist()

train_df = train_df.drop(index=ind_list)
data = pd.concat([train_df.SalePrice, train_df.TotalBsmtSF], axis = 1)

data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))
# Taking out the clear outlier in TotalBsmtSF parameter

ind_list = train_df.index[train_df['TotalBsmtSF'].values > 5000].tolist()

train_df = train_df.drop(index=ind_list)
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
#correlation matrix

corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
k=11

cols = corrmat.nlargest(k, 'SalePrice').index

cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.0)

fig, ax = plt.subplots(figsize=(10,10))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='0.2f', annot_kws={'size': 12},

                 yticklabels=cols.values, xticklabels=cols.values, ax=ax)

plt.show()
# train_df = train_df.drop(train_df[(train_df['OverallQual']<5) & (train_df['SalePrice']>200000)].index, inplace=True)

# train_df = train_df.drop(train_df[(train_df['GrLivArea']>4500) & (train_df['SalePrice']<250000)].index, inplace=True)

# train_df = train_df.reset_index(drop=True, inplace=True)



ind_list = train_df.index[(train_df['OverallQual'].values <= 5) & (train_df['SalePrice'].values > 200000)].tolist()

train_df = train_df.drop(train_df.index[ind_list])

ind_list = train_df.index[(train_df['GrLivArea'].values >= 4500) & (train_df['SalePrice'].values < 250000)].tolist()

train_df = train_df.drop(train_df.index[ind_list])
# split features and Labels



train_labels_series  = train_df.SalePrice.reset_index(drop=True)

train_features_df = train_df.drop('SalePrice', axis=1)

test_features_df = test_df



# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset

all_features_df = pd.concat([train_features_df, test_features_df]).reset_index(drop=True)

all_features_df = all_features_df.drop('Id', axis=1)

all_features_df.shape

# missing data

total = all_features_df.isnull().sum().sort_values(ascending=False)

percent = (all_features_df.isnull().sum()/all_features_df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
drop_index = missing_data[missing_data['Percent'] > 0.15].index

all_features_df = all_features_df.drop(drop_index, axis=1)

all_features_df = all_features_df.drop(['GarageCond', 'GarageQual', 'GarageYrBlt', 'BsmtQual', 'BsmtCond'], axis = 1)
all_features_df = all_features_df.fillna(method='ffill')

all_features_df = all_features_df.drop(['3SsnPorch'], axis=1)
# train_df.SalePrice = log1p(train_df.SalePrice)

sns.distplot(train_labels_series)
train_labels_series = np.log1p(train_labels_series)

sns.distplot(train_labels_series)
all_features_df.MSSubClass = all_features_df.MSSubClass.astype(str)

all_features_df.YrSold = all_features_df.YrSold.astype(str)

all_features_df.MoSold = all_features_df.MoSold.astype(str)

# fetch all numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric_cols = []



for col in all_features_df.columns:

    if all_features_df[col].dtype in numeric_dtypes:

        numeric_cols.append(col)



print(numeric_cols)
# Create box plots for all numeric features

sns.set_style("white")

f, ax = plt.subplots(figsize=(12, 8))

ax.set_xscale("log")

ax = sns.boxplot(data=all_features_df[numeric_cols] , orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
# skewness of all numeric columns

skewness = []



for col in numeric_cols:

    skew = all_features_df[col].skew()

    skewness.append(skew)



skew_df = pd.Series(skewness, index = numeric_cols).sort_values(ascending = False)

skew_df.head(20)
# select only the highskew features



skew_df = skew_df[skew_df>0.75]

print('The features with high skewness are given below' )

skew_df
# Normalie skewed features using Box-Cox Transformation

all_features_df_future_use = all_features_df



for i in skew_df.index:

    all_features_df[i] = boxcox1p(all_features_df[i], boxcox_normmax(all_features_df[i] + 1))

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

ax.set_xscale("log")

ax = sns.boxplot(data=all_features_df[skew_df.index] , orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
# Fetch all numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = []

for i in all_features_df.columns:

    if all_features_df[i].dtype in numeric_dtypes:

        numeric.append(i)

        
scaler = StandardScaler()

all_numeric_features_df = all_features_df[numeric]

all_categoric_features_df = all_features_df.drop(columns = numeric)

numeric_columns = all_numeric_features_df.columns

all_numeric_features = scaler.fit_transform(all_numeric_features_df)
all_categoric_features_df = pd.get_dummies(all_categoric_features_df).reset_index(drop=True)

all_categoric_features_df = all_categoric_features_df.loc[:,~all_categoric_features_df.columns.duplicated()]
all_numeric_features_df = pd.DataFrame(data=all_numeric_features, columns=numeric_columns)

all_features_transformed_df = pd.concat([all_numeric_features_df, all_categoric_features_df], axis = 1)
X_train = all_features_transformed_df.iloc[:len(train_labels_series)]

X_test = all_features_transformed_df.iloc[len(train_labels_series):]
X_train_new, X_cross_val, Y_train_new, Y_cross_val = train_test_split(X_train, train_labels_series, test_size = 0.1,

                                                             random_state = 0)
# Define error metric

def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))





alphas = np.logspace(-4, 6, 20)

train_rmse_list = []

cv_rmse_list = []

# Choosing a simple regressor with Regularization

for alpha in alphas:

    regressor = Ridge(alpha=alpha)

    regressor.fit(X_train_new, Y_train_new)

    y_pred_train = regressor.predict(X_train_new)

    y_pred_cv = regressor.predict(X_cross_val)

    train_error = rmse(Y_train_new, y_pred_train)

    cv_error = rmse(Y_cross_val, y_pred_cv)

    train_rmse_list.append(train_error)

    cv_rmse_list.append(cv_error)

alpha_log = np.log(alphas)

plt.plot(alpha_log, train_rmse_list, 'b')

plt.plot(alpha_log, cv_rmse_list, 'g')

plt.title('Root Mean squared error')

plt.xlabel('Log (correl coef)')

plt.ylabel('RMSE')

plt.show()
columns = all_features_df.columns



for col in columns:

    print(col)

    print(np.unique(all_features_df[col].values))
all_features_transformed_df['Has_BsmtFinType1_Unf'] = 1* (all_features_df['BsmtFinType1'] == 'Unf')

all_features_transformed_df['Has_Severe_Land_Slope'] = 1* (all_features_df['LandSlope'] == 'Sev')

# Next feature is to consider some heritage value if present

all_features_transformed_df['Is_heritage'] = 1 * (all_features_df['YearBuilt'] <= 1900)

all_features_transformed_df['Has_poor_heating'] = 1 * (all_features_df['HeatingQC'] == 'Po')

all_features_transformed_df['Has_Basement'] = 1 * (all_features_df['TotalBsmtSF'] > 0)

all_features_transformed_df['Has_Central_Air'] = 1 * (all_features_df['CentralAir'] == 'Y')

all_features_transformed_df['Has_second_floor'] = 1 * (all_features_df['2ndFlrSF'] > 0)

all_features_transformed_df['Has_Bath'] = 1 * (all_features_df['FullBath'] > 0)

all_features_transformed_df['Has_Br_above_Grade'] = 1 * (all_features_df['BedroomAbvGr'] > 0)

YearsSinceRemodel = all_features_df['YrSold'].astype(int) - all_features_df['YearRemodAdd'].astype(int)

YearsSinceRemodel = YearsSinceRemodel.values.reshape(-1, 1)

all_features_transformed_df['YearsSinceRemodel'] = scaler.fit_transform(YearsSinceRemodel)

all_features_transformed_df['Has_Garage'] = 1 * (all_features_df['GarageCars'] > 0)

all_features_transformed_df['Has_wood_deck'] = 1 * (all_features_df['WoodDeckSF'] > 0)

all_features_transformed_df['Has_open_porch'] = 1 * (all_features_df.OpenPorchSF > 0)

all_features_transformed_df['Has_open_porch'] = 1 * (all_features_df.ScreenPorch > 0)

all_features_transformed_df['Has_pool'] = 1 * (all_features_df.PoolArea > 0)

X_train = all_features_transformed_df.iloc[:len(train_labels_series)]

X_test = all_features_transformed_df.iloc[len(train_labels_series):]

X_train_new, X_cross_val, Y_train_new, Y_cross_val = train_test_split(X_train, train_labels_series, test_size = 0.1,

                                                             random_state = 20)

alphas = np.logspace(-4, 6, 20)

train_rmse_list = []

cv_rmse_list = []

# Choosing a simple regressor with Regularization

for alpha in alphas:

    regressor = Ridge(alpha=alpha)

    regressor.fit(X_train_new, Y_train_new)

    y_pred_train = regressor.predict(X_train_new)

    y_pred_cv = regressor.predict(X_cross_val)

    train_error = rmse(Y_train_new, y_pred_train)

    cv_error = rmse(Y_cross_val, y_pred_cv)

    train_rmse_list.append(train_error)

    cv_rmse_list.append(cv_error)

    

alpha_log = np.log(alphas)

plt.plot(alpha_log, train_rmse_list, 'b')

plt.plot(alpha_log, cv_rmse_list, 'g')

plt.title('Root Mean squared error')

plt.xlabel('Log (correl coef)')

plt.ylabel('RMSE')

plt.show()
# cross validation folds setup

kf = KFold(n_splits=10, shuffle=True)



def cross_val_rmse(model):

    rmse = np.sqrt(cross_val_score(model, X=X_train, y=train_labels_series , cv=kf))

    return rmse
# Ridge Regressor

alphas = np.logspace(-4, 6, 20)

ridge_reg = RidgeCV(alphas=alphas, cv=kf)



# Gradient Boosting Regressor

grad_boost_reg = GradientBoostingRegressor(loss = 'huber', n_estimators = 3000, learning_rate = 0.01,

                                       max_depth = 5, max_features = 'sqrt', min_samples_split=10,

                                      min_samples_leaf=15, random_state=20)



# Random forest Regressor

rf_reg = RandomForestRegressor(n_estimators=3000, max_depth=10, min_samples_split=5, min_samples_leaf=7,

                          max_features=None, oob_score=True, random_state=42)



# Support Vector Regressor

sv_reg = SVR(C= 10, epsilon= 0.01, gamma='scale')



# XG Boost Regressor

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3000, max_depth=4, min_child_weight=0,

                       gamma=0.6,subsample=0.7, colsample_bytree=0.7, objective='reg:linear', nthread=-1,

                       scale_pos_weight=1, seed=27, reg_alpha=0.00006, random_state=20)





# Define error metric

def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



model = ['Ridge', 'Gradient Boost', 'Random Forest', 'Support Vector', 'XG Boost']

errors = []



# Ridge Regressor

regressor = ridge_reg.fit(X_train, train_labels_series)

y_pred_train = regressor.predict(X_train)

error = rmse(train_labels_series, y_pred_train)

errors.append(error)

print('Ridge Regressor RMSE: %s' % error)



# Gradient Boosting Regressor

regressor = grad_boost_reg.fit(X_train, train_labels_series)

y_pred_train = regressor.predict(X_train)

error = rmse(train_labels_series, y_pred_train)

errors.append(error)

print('Gradient Boosting Regressor RMSE: %s' % error)



#  Random forest Regressor

regressor = rf_reg.fit(X_train, train_labels_series)

y_pred_train = regressor.predict(X_train)

error = rmse(train_labels_series, y_pred_train)

errors.append(error)

print('Random forest Regressor RMSE: %s' % error)



# Support Vector Regressor

regressor = sv_reg.fit(X_train, train_labels_series)

y_pred_train = regressor.predict(X_train)

error = rmse(train_labels_series, y_pred_train)

errors.append(error)

print('Support Vector Regressor RMSE: %s' % error)



# XG Boost Regressor

regressor = xgboost.fit(X_train, train_labels_series)

y_pred_train = regressor.predict(X_train)

error = rmse(train_labels_series, y_pred_train)

errors.append(error)

print('XG Boost Regressor RMSE: %s' % error)

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(model,errors)

plt.xticks(model, rotation=30)

plt.title('Root mean squared error for different models')

plt.show()
# Usin Support Vector Regressor

sv_reg = SVR(C= 15, epsilon= 0.01, gamma='scale')

X_train_new, X_cross_val, Y_train_new, Y_cross_val = train_test_split(X_train, train_labels_series, test_size = 0.1,

                                                             random_state = 30)



regressor = sv_reg.fit(X_train_new, Y_train_new)

y_pred_train = regressor.predict(X_train_new)

y_pred_cv = regressor.predict(X_cross_val)

error_train = rmse(Y_train_new, y_pred_train)

error_cv = rmse(Y_cross_val, y_pred_cv)

print('Support Vector Regressor train error: %s' % error_train)

print('Support Vector Regressor CV error: %s' % error_cv)
Y_pred_test = regressor.predict(X_test)

# Transforming the predictions

Y_pred_test = np.expm1(Y_pred_test)



submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.SalePrice = Y_pred_test
submission.to_csv("submission_1.csv", index=False)