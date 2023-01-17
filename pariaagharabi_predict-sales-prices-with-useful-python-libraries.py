# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None



import warnings

warnings.simplefilter(action ="ignore")



import matplotlib.pyplot as plt

import seaborn as sns



from scipy import stats

import math



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

import sklearn.model_selection as GridSearchCV

import sklearn.model_selection as ms

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNetCV

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.ensemble import VotingRegressor

from mlxtend.regressor import StackingCVRegressor

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
test.head()
# Analyse statically insight of train data

train.describe()
# Analyse statically insight of test data

test.describe()
train.info()
test.info()
print(f"The train data size: {train.shape}")

print(f"The test data size: {test.shape}")
diff_train_test = set(train.columns) - set(test.columns)

diff_train_test
train["SalePrice"].describe()
print(f"Skewness of SalePrice: {train['SalePrice'].skew()}")

print(f"Kurtosis of SalePrice: {train['SalePrice'].kurt()}")
# Plot a histogram and kernel density estimate for SalePrice target

sns.distplot(train["SalePrice"], color = "#330033");

plt.xlabel("Sale price", fontsize = 14, color = "#330033" );
sns.distplot(np.log1p(train["SalePrice"]));
ax = sns.distplot(train["SalePrice"], bins=20, kde=False, fit=stats.norm);

plt.title("Distribution of SalePrice")



# Get the fitted parameters used by sns

(mu, sigma) = stats.norm.fit(train["SalePrice"])

print("mu={:.2f}, sigma={:.2f}".format(mu, sigma))



# Legend and labels 

plt.legend(["Normal dist. fit ($\mu=${:.2f}, $\sigma=${:.2f})".format(mu, sigma)])

plt.ylabel("Frequency")



# Cross-check this is indeed the case - should be overlaid over black curve

x_dummy = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)

ax.plot(x_dummy, stats.norm.pdf(x_dummy, mu, sigma))

plt.legend(["Normal dist. fit ($\mu=${:.2f}, $\sigma=${:.2f})".format(mu, sigma), "cross-check"]);
numeric_features = train.select_dtypes(include=[np.number])

corr_numeric_features = numeric_features.corr()
# Correlation Numeric featurs with output variable(SalePrice)

corr_target = abs(corr_numeric_features["SalePrice"])

print(f"Correlation between numeric featurs and SalePrice:\n{corr_target.sort_values()}")
relevant_features = corr_target[corr_target > 0.6]

relevant_features

print(f"Selecting highly correlated numeric features with SalePrice:\n{relevant_features}")
highly_correlated_visualization = sns.pairplot(train[["OverallQual",  "TotalBsmtSF", "1stFlrSF", "GrLivArea", "GarageCars", "GarageArea", "SalePrice"]])
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train["SalePrice"].to_frame()



#Combine train and test sets

concat_data = pd.concat((train, test), sort=False).reset_index(drop=True)

#Drop the target "SalePrice" and Id columns

concat_data.drop(["SalePrice"], axis=1, inplace=True)

concat_data.drop(["Id"], axis=1, inplace=True)

print("Total size is :",concat_data.shape)
concat_data.head()
concat_data.info()
# Count the null columns

null_columns = concat_data.columns[concat_data.isnull().any()]

concat_data[null_columns].isnull().sum()
df = concat_data.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu"], axis = 1) 
print(f"The full data size: {df.shape}")
# Count the null columns

null_columns = df.columns[df.isnull().any()]

df[null_columns].isnull().sum()
numeric_features = df.select_dtypes(include=[np.number])

numeric_features.dtypes
print(f"Numerical features: {numeric_features.shape}")
unique_list_numeric_features = [(item, np.count_nonzero(df[item].unique())) for item in numeric_features]

print(f"Unique numeric features:\n{unique_list_numeric_features}")
# Corralation between Numeric features 

corr_numeric_features = numeric_features.corr()



#Using Pearson Correlation

plt.figure(figsize=(30, 30))



sns.heatmap(corr_numeric_features, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap="Blues")



plt.show()
# Count the null columns' numeric_features in data set

null_columns_numeric_features = numeric_features.columns[numeric_features.isnull().any()]

print(f"Missing values in numerical features: \n{numeric_features[null_columns_numeric_features].isnull().sum()}")
cols_absence_zero = ["LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF","TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]

df[cols_absence_zero] = df[cols_absence_zero].replace(to_replace = np.nan, value = 0) 
df["GarageYrBlt"] = df.apply(

    lambda row: row["YearBuilt"] if np.isnan(row["GarageYrBlt"]) else row["GarageYrBlt"],

    axis=1

)
def ReplaceNanWithMedian(df, featureName):

    median = df.loc[:,featureName].median()



    df[featureName] = df.apply(lambda row: median if np.isnan(row[featureName]) else row[featureName], axis=1)
ReplaceNanWithMedian(df, "GarageCars")

ReplaceNanWithMedian(df, "GarageArea")
categoricals = df.select_dtypes(exclude=[np.number])

categoricals.dtypes
print(f"Categorical features: {categoricals.shape}")
 # Count the null columns

null_columns = categoricals.columns[categoricals.isnull().any()]

print(f"Missing values in categorical features: \n{categoricals[null_columns].isnull().sum()}")
cols_absence_none = ["MasVnrType", "BsmtQual", "BsmtExposure","BsmtFinType1", "BsmtFinType2","GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtCond"]

df[cols_absence_none] = df[cols_absence_none].replace(to_replace = np.nan, value = "None") 
def ReplaceNanWithMostFrequent(df, featureName):

    df[featureName] = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
ReplaceNanWithMostFrequent(df, "MSZoning")

ReplaceNanWithMostFrequent(df, "Utilities")

ReplaceNanWithMostFrequent(df, "Exterior1st")

ReplaceNanWithMostFrequent(df, "Exterior2nd")

ReplaceNanWithMostFrequent(df, "MasVnrType")

ReplaceNanWithMostFrequent(df, "Electrical")

ReplaceNanWithMostFrequent(df, "KitchenQual")

ReplaceNanWithMostFrequent(df, "Functional")

ReplaceNanWithMostFrequent(df, "SaleType")
 # Check the null columns in data set

null_columns = df.columns[df.isnull().any()]

df[null_columns].isnull().sum()
# Check if the all of the columns have 0 null values.

sum(df.isnull().sum() != 0)
print(f"The Shape of all data: {df.shape}")
# Transforming some numerical variables that are really categorical

df["MSSubClass"] = df["MSSubClass"].apply(str)



#Changing OverallCond into a categorical variable

df["OverallCond"] = df["OverallCond"].astype(str)



#Year and month sold are transformed into categorical features.

df["YrSold"] = df["YrSold"].astype(str)

df["MoSold"] = df["MoSold"].astype(str)
final_df = pd.get_dummies(df).reset_index(drop=True)
print(f"Original dataset shape: {df.shape}")

print(f"Encoded dataset shape: {final_df.shape}")

print(f"We have: {final_df.shape[1] - df.shape[1]} new encoded features")
TrainData = final_df[:ntrain] 

TestData = final_df[ntrain:]
TrainData.shape, TestData.shape
fig, ax_arr = plt.subplots(3, 2, figsize=(14, 14))



ax_arr[0, 0].scatter(x = train["OverallQual"], y = train["SalePrice"], color="#330033", alpha=.3)

ax_arr[0, 0].set_title("House price Vs Overall material and finish quality", fontsize=14, color="#330033")



ax_arr[0, 1].scatter(x = train["TotalBsmtSF"], y = train["SalePrice"], color="#330033", alpha=.3)

ax_arr[0, 1].set_title("House price Vs Total square feet of basement area", fontsize=14, color="#330033")



ax_arr[1, 0].scatter(x = train["1stFlrSF"], y = train["SalePrice"], color="#330033", alpha=.3)

ax_arr[1, 0].set_title("House price Vs First Floor square feet", fontsize=14, color="#330033")



ax_arr[1, 1].scatter(x = train["GrLivArea"], y = train["SalePrice"], color="#330033", alpha=.3)

ax_arr[1, 1].set_title("House price Vs Above grade (ground) living area square feet", fontsize=14, color="#330033")



ax_arr[2, 0].scatter(x = train["LotArea"], y = train["SalePrice"], color="#330033", alpha=.3)

ax_arr[2, 0].set_title("House price Vs Lot size in square feet", fontsize=14, color="#330033")



ax_arr[2, 1].scatter(x = train["YearBuilt"], y = train["SalePrice"], color="#330033", alpha=.3)

ax_arr[2, 1].set_title("House price Vs Original construction date", fontsize=14, color="#330033")



plt.tight_layout()



plt.show()
OverallQual_visualization = sns.swarmplot(y = "SalePrice", x = "OverallQual", data = train, size = 7)

# remove the top and right line in graph

sns.despine()

OverallQual_visualization.figure.set_size_inches(14,10)

plt.show()
TotalBsmtSF_visualization = sns.swarmplot(y = "SalePrice", x = "TotalBsmtSF", data = train, size = 7)

# remove the top and right line in graph

sns.despine()

TotalBsmtSF_visualization.figure.set_size_inches(14,10)

plt.show()
StFlrSF_visualization = sns.swarmplot(y = "SalePrice", x = "1stFlrSF", data = train, size = 7)

# remove the top and right line in graph

sns.despine()

StFlrSF_visualization.figure.set_size_inches(14,10)

plt.show()
LivArea_visualization = sns.swarmplot(y = "SalePrice", x = "GrLivArea", data = train, size = 7)

# remove the top and right line in graph

sns.despine()

LivArea_visualization.figure.set_size_inches(14,10)

plt.show()
LotArea_visualization = sns.swarmplot(y = "SalePrice", x = "LotArea", data = train, size = 7)

# remove the top and right line in graph

sns.despine()

LotArea_visualization.figure.set_size_inches(14,10)

plt.show()
YearBuilt_visualization = sns.swarmplot(y = "SalePrice", x = "YearBuilt", data = train, size = 7)

# remove the top and right line in graph

sns.despine()

YearBuilt_visualization.figure.set_size_inches(14,10)

plt.show()
MasVnrArea_visualization = sns.swarmplot(y = "SalePrice", x = "MasVnrArea", data = train, size = 7)

# remove the top and right line in graph

sns.despine()

MasVnrArea_visualization.figure.set_size_inches(14,10)

plt.show()
train_df = TrainData[(TrainData["GrLivArea"] < 4600) & (TrainData["MasVnrArea"] < 1500)]

print(f"We removed: {TrainData.shape[0]- train_df.shape[0]} outliers")
print(f"Encoded dataset shape: {final_df.shape}")
target = train[["SalePrice"]]
pos = [1298,523, 297]

target.drop(target.index[pos], inplace=True)
print("We make sure that both train and target sets have the same row number after removing the outliers:")

print(f"Train: {train_df.shape[0]} rows")

print(f"Target: {target.shape[0]} rows")
print(f"Skewness before log transform: {target.SalePrice.skew()}")

print(f"Kurtosis before log transform: {target.SalePrice.kurt()}")
target["SalePrice"] = np.log1p(target["SalePrice"])
print(f"Skewness after log transform: {target.SalePrice.skew()}")

print(f"Kurtosis after log transform: {target.SalePrice.kurt()}")
final_df = final_df.loc[:,~final_df.columns.duplicated()]
x = train_df

y = np.array(target)
# Split the data set into train and test sets 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
scaler = RobustScaler()
# transform "x_train"

x_train = scaler.fit_transform(x_train)

# transform "x_test"

x_test = scaler.transform(x_test)

#Transform the test set

X_test= scaler.transform(TestData)
ridge = Ridge()

parameters = {"alpha":[x for x in range(1,101)]}

ridge_regressor = ms.GridSearchCV(ridge, param_grid = parameters, scoring = "neg_mean_squared_error", cv = 15)

ridge_regressor_mod = ridge_regressor.fit(x_train, y_train)

print(f"Best parameter for Ridge regression: {ridge_regressor_mod.best_params_}")
ridge = Ridge(alpha = 13)

ridge_mod = ridge.fit(x_train, y_train)

pred_ridge = ridge_mod.predict(x_test) 

mse_ridge = mean_squared_error(y_test,pred_ridge)

rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))

score_ridge_train = ridge_mod.score(x_train, y_train)

score_ridge_test = ridge_mod.score(x_test, y_test)
print(f"Mean Square Error for Ridge regression = {mse_ridge}")

print(f"Root Mean Square Error for Ridge regression = {rmse_ridge}")

print(f"R^2(coefficient of determination) on training set = {score_ridge_train}")

print(f"R^2(coefficient of determination) on testing set = {score_ridge_test}")
# Print coefficients

print(f"Ridge coefficient:\n {ridge.coef_}") 
parameters= {"alpha":[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}



lasso = Lasso()

lasso_reg = ms.GridSearchCV(lasso, param_grid = parameters, scoring = "neg_mean_squared_error", cv = 15)

lasso_reg.fit(x_train,y_train)



print("The best value of Alpha for Lasso regression is: ",lasso_reg.best_params_)
lasso = Lasso(alpha = 0.0009)

lasso_mod = lasso.fit(x_train, y_train)

pred_lasso = lasso_mod.predict(x_test) 



mse_lasso = mean_squared_error(y_test,pred_lasso)

rmse_lasso = np.sqrt(mean_squared_error(y_test, pred_lasso))

score_lasso_train = lasso_mod.score(x_train, y_train)

score_lasso_test = lasso_mod.score(x_test, y_test)
print(f"Mean Square Error for Lasso regression = {mse_lasso}")

print(f"Root Mean Square Error for Lasso regression = {rmse_lasso}")

print(f"R^2(coefficient of determination) on training set = {score_lasso_train}")

print(f"R^2(coefficient of determination) on testing set = {score_lasso_test}")
# Print coefficients

print(f"Lasso coefficient:\n {lasso.coef_}")
coefs = pd.Series(lasso_mod.coef_, index = x.columns)

plt.figure(figsize=(20, 20))



imp_coefs = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh", color = "#800080")

plt.xlabel("Lasso coefficient", weight = "bold")

plt.title("Feature importance in the Lasso Model", weight = "bold", color = "#800080")

plt.show()
print("Lasso kept ",sum(coefs != 0), "important features and dropped the other ", sum(coefs == 0)," features")
alphas = [0.0005]

l1ratio = [0.9]



elastic_net_cv = ElasticNetCV(cv = 5, max_iter = 1e7, alphas = alphas,  l1_ratio = l1ratio)

elastic_mod = elastic_net_cv.fit(x_train, y_train.ravel())

pred_elastic = elastic_mod.predict(x_test) 



mse_elastic = mean_squared_error(y_test, pred_elastic)

rmse_elastic = np.sqrt(mean_squared_error(y_test, pred_elastic))

score_elastic_train = elastic_mod.score(x_train, y_train)

score_elastic_test = elastic_mod.score(x_test, y_test)
print(f"Mean Square Error for Elastic Net CV regression = {mse_elastic}")

print(f"Root Mean Square Error for Elastic Net CV regression = {rmse_elastic}")

print(f"R^2(coefficient of determination) on training set = {score_elastic_train}")

print(f"R^2(coefficient of determination) on testing set = {score_elastic_test}")
# Print coefficients

print(f"Elastic Net CV coefficient:\n {lasso.coef_}")
svr = SVR(C = 20, epsilon = 0.008, gamma = 0.0003,)

svr_mod = svr.fit(x_train, y_train.ravel())

pred_svr = svr_mod.predict(x_test) 



mse_svr = mean_squared_error(y_test, pred_svr)

rmse_svr = np.sqrt(mean_squared_error(y_test, pred_svr))

score_svr_train = svr_mod.score(x_train, y_train)

score_svr_test = svr_mod.score(x_test, y_test)
print(f"Mean Square Error for SVR = {mse_svr}")

print(f"Root Mean Square Error for SVR = {rmse_svr}")

print(f"R^2(coefficient of determination) on training set = {score_svr_train}")

print(f"R^2(coefficient of determination) on testing set = {score_svr_test}")
gbr = GradientBoostingRegressor(n_estimators = 3000, learning_rate = 0.05, max_depth = 4, max_features = "sqrt", min_samples_leaf = 15, min_samples_split = 10, loss = "huber", random_state = 42)

gbr_mod = gbr.fit(x_train, y_train.ravel())

pred_gbr = gbr_mod.predict(x_test) 



mse_gbr = mean_squared_error(y_test, pred_gbr)

rmse_gbr = np.sqrt(mean_squared_error(y_test, pred_gbr))

score_gbr_train = gbr_mod.score(x_train, y_train)

score_gbr_test = gbr_mod.score(x_test, y_test)
print(f"Mean Square Error for Gradient Boosting regression = {mse_gbr}")

print(f"Root Mean Square Error for Gradient Boosting regression = {rmse_gbr}")

print(f"R^2(coefficient of determination) on training set = {score_gbr_train}")

print(f"R^2(coefficient of determination) on testing set = {score_gbr_test}")
lightgbm = LGBMRegressor(objective = "regression", 

                                       num_leaves = 4,

                                       learning_rate = 0.01, 

                                       n_estimators = 5000,

                                       max_bin = 200, 

                                       bagging_fraction = 0.75,

                                       bagging_freq = 5, 

                                       bagging_seed = 7,

                                       feature_fraction = 0.2,

                                       feature_fraction_seed = 7,

                                       verbose = -1,

                                       )

lightgbm_mod = lightgbm.fit(x_train, y_train.ravel())

pred_lightgbm = lightgbm_mod.predict(x_test) 



mse_lightgbm = mean_squared_error(y_test, pred_lightgbm)

rmse_lightgbm = np.sqrt(mean_squared_error(y_test, pred_lightgbm))

score_lightgbm_train = lightgbm_mod.score(x_train, y_train)

score_lightgbm_test = lightgbm_mod.score(x_test, y_test)
print(f"Mean Square Error for LGBMRegressor = {mse_lightgbm}")

print(f"Root Mean Square Error for LGBMRegressor = {rmse_lightgbm}")

print(f"R^2(coefficient of determination) on training set = {score_lightgbm_train}")

print(f"R^2(coefficient of determination) on testing set = {score_lightgbm_test}")
xgboost = XGBRegressor(learning_rate = 0.01, n_estimators = 3460,

                                     max_depth = 3, min_child_weight = 0,

                                     gamma = 0, subsample = 0.7,

                                     colsample_bytree = 0.7,

                                     objective = "reg:squarederror", nthread=-1,

                                     scale_pos_weight = 1, seed = 27,

                                     reg_alpha = 0.00006)

xgboost_mod = xgboost.fit(x_train, y_train)

pred_xgboost = xgboost_mod.predict(x_test) 



mse_xgboost = mean_squared_error(y_test, pred_xgboost)

rmse_xgboost = np.sqrt(mean_squared_error(y_test, pred_xgboost))

score_xgboost_train = xgboost_mod.score(x_train, y_train)

score_xgboost_test = xgboost_mod.score(x_test, y_test)
print(f"Mean Square Error for xgboost regression = {mse_lightgbm}")

print(f"Root Mean Square Error for xgboost regression = {rmse_lightgbm}")

print(f"R^2(coefficient of determination) on training set = {score_lightgbm_train}")

print(f"R^2(coefficient of determination) on testing set = {score_lightgbm_test}")
vote = VotingRegressor([("Ridge", ridge_mod), ("Lasso", ridge_mod), ("Elastic Net CV", elastic_net_cv), 

                        ("SVR", svr), ("GradientBoostingRegressor", gbr), ("LGBMRegressor", lightgbm), ("XGBRegressor", xgboost)])

vote_mod = vote.fit(x_train, y_train.ravel())

vote_pred = vote_mod.predict(x_test)



print(f"Root Mean Square Error test for ENSEMBLE METHODS: {np.sqrt(mean_squared_error(y_test, vote_pred))}")
stack_gen = StackingCVRegressor(regressors = [ridge_mod, ridge_mod, elastic_net_cv, svr, gbr, lightgbm, xgboost, vote],

                                meta_regressor = xgboost,

                                use_features_in_secondary = True)

stack_mod = stack_gen.fit(x_train, y_train.ravel())

stack_pred = stack_mod.predict(x_test)



print(f"Root Mean Square Error test for STACKING REGRESSOR: {np.sqrt(mean_squared_error(y_test, vote_pred))}")
averaged_preds = (vote_pred*0.3 + stack_pred*0.5 + pred_lasso*0.2)



print(f"Root Mean Square Error test for STACKING REGRESSOR: {np.sqrt(mean_squared_error(y_test, averaged_preds))}")
# Visualize the result in a plot with averaging predict.



plt.figure(figsize=(20,8))



x_ax = range(len(x_test))

plt.scatter(x_ax, y_test, s = 5, color="#422d42", label = "Original")

plt.plot(x_ax, averaged_preds, lw = 0.8, color = "#9c6d9c", label = "Predicted")

plt.legend()

plt.show()
#VotingRegressor to predict the final Test

vote_test = vote.predict(X_test)

final1 = np.expm1(vote_test)



#StackingRegressor to predict the final Test

stack_test = stack_gen.predict(X_test)

final2 = np.expm1(stack_test)



#LassoRegressor to predict the final Test

lasso_test = lasso.predict(X_test)

final3 = np.expm1(lasso_test)
test["Id"].value_counts()
final = (0.2*final1 + 0.6*final2 + 0.2*final3)



final_submission = pd.DataFrame({"Id": test["Id"], "SalePrice": final})

final_submission.to_csv("final_submission.csv", index=False)

final_submission.head()