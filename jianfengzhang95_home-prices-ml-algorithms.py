import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.impute import SimpleImputer

from scipy.stats import skew 
## 
## modeling Algorithms
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from scipy.special import boxcox1p
# Read data and briefly view of Train dataset
dt_train = pd.read_csv("train.csv")
dt_test = pd.read_csv("test.csv")
dt_train.head(5)

## Dimension of the train dataset
print(dt_train.shape)
# print(dt_test.shape)
row_num_train = dt_train.shape[0]
row_num_test = dt_test.shape[0]
print(dt_train.info())
## Missing Value percent of the train dataset
missing_val_train_col = (dt_train.isnull().sum())
missing_val_train_col_percent = missing_val_train_col / dt_train.shape[0]
missing_val_train_col_percent[ missing_val_train_col_percent > 0 ].sort_values(ascending=False)

missing_val_train_col[missing_val_train_col > 0]
missing_val_train_col_percent[ missing_val_train_col_percent > 0.8 ]
## Missing Value percent of the test dataset
missing_val_test_col = (dt_test.isnull().sum())
missing_val_test_col_percent = missing_val_test_col / dt_test.shape[0]
missing_val_test_col_percent[ missing_val_test_col_percent > 0.8 ].sort_values(ascending=False)

#missing_val_test_col[missing_val_train_col > 0]
## Summary of the variables.
dt_train.describe()
## The distribution of the "SalePrice"
plt.hist(dt_train["SalePrice"], normed = True, bins = 50,histtype = "stepfilled")
plt.ylabel('probability')
plt.xlabel('value of sale price')

plt.title("histgram of sales price")
sns.set(color_codes = True)
sns.distplot( dt_train["SalePrice"] )
dt_train["SalePrice"].describe()
## Get the numerical and categorical variables.
numerical_features =  dt_train.select_dtypes(include=[np.number])
numerical_columns = numerical_features.columns.tolist()

categorical_features = dt_train.select_dtypes(include=[np.object])
categorical_columns = categorical_features.columns.tolist()


print(len(numerical_columns))

print(sorted(numerical_columns))
print(sorted(categorical_columns))

## Pick important variables that can be applied to build heat map/ The variables used donot have any Missing Values
# Using decision tree classifier 

def plot_variable_importance( X, y):
    tree = DecisionTreeClassifier ( random_state = 99)
    tree.fit(X, y)
    plot_model_var_imp (tree, X, y)
    




train_numerical = dt_train[numerical_columns]
train_x_numerical = dt_train[ numerical_columns[:-1] ]
train_y_numerical = dt_train[ numerical_columns[-1] ]


train_x_numerical.head()
train_categorical = dt_train[categorical_columns]

train_categorical.head()
# The histgrams of numerical variables.
train_numerical.hist(figsize=(15,20))
plt.figure()








# plt.boxplot(train_categorical)




# plot_variable_importance(train_x_numerical, train_y_numerical)
# plt.show()

## Part 1: analyse the numerical variables through multiple plots
# Plot 1: heatmap of numerical variables

plt.figure(figsize=(7,4)) 

sns.heatmap(dt_train[numerical_columns].corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()





test1 = pd.concat((dt_train, dt_test)).reset_index(drop=True)
test1.shape
target_train = dt_train["SalePrice"]
all_data = pd.concat((dt_train, dt_test)).reset_index(drop=True)
print(all_data.shape)

all_data.head()
print(skew(all_data["MiscVal"]))
t = all_data["MiscVal"]
lam1 = 0.1
t2 = boxcox1p( t, 0.15)
skew(t2)

skew(t)
# drop the column ID and Target value: "SalePrice"
all_data.drop(['Id'], axis=1, inplace=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.shape

## Missing Value check

#
missing_val_col= (all_data.isnull().sum())
missing_val_col_percent = missing_val_col / all_data.shape[0]
missing_percent = missing_val_col_percent[ missing_val_col_percent > 0 ].sort_values(ascending=False)
missing_percent

f, ax = plt.subplots(figsize=(12, 9))
sns.barplot(x = missing_percent.index, y= missing_percent * 100)
plt.xticks(rotation='90')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
# Get the missing values for categorical variables.

all_data_numerical_features =  all_data.select_dtypes(include=[np.number])
all_data_numerical_columns = all_data_numerical_features.columns.tolist()


all_data_categorical_features = all_data.select_dtypes(include=[np.object])
all_data_categorical_columns = all_data_categorical_features.columns.tolist()

categorical_missing = []
for mis in missing_percent.index:
    if mis in all_data_categorical_columns:
        categorical_missing.append(mis)
print(len(categorical_missing))
categorical_missing
numerical_missing = [ ]
for mis in missing_percent.index:
    if mis not in categorical_missing:
        numerical_missing.append(mis)
      
print(len(numerical_missing))
numerical_missing  
## Impute categorical vairbales. The missing value of categorical variable 

# 1. 
# Impute the PoolQC: 
# this variable represents the quality of pool. Comparing the variable "PoolArea", it could be easy to find
# that the NA value in PoolQC means there is no pool for the house. When "poolQC" equals to 0, the value of "PoolArea" is 0.
# So I decided to impute this feature by "None" 
pd.Series(all_data["PoolQC"]).value_counts()
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
# 2. Impute the MiscFeature:
# this variable represents the Miscellaneous feature not covered in other categories.Comparing the variable "MiscVal", 
# it could be easy to find that the NA value in MiscFeature means there is no value for Miscellaneous feature.
# When "MiscFeature" equals to 0, the value of "MiscVal" is 0. So I decided to impute this feature by "None" 
print(pd.Series(all_data["MiscFeature"]).value_counts())
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("MiscFeature")
# 3. Impute the Alley:
# this variable represents the type of alley access. When "Alley" equals to 0, it means that there is no alley.
# So I decided to impute this feature by "None" 

print(pd.Series(all_data["Alley"]).value_counts())
all_data["Alley"] = all_data["Alley"].fillna("None")

# 4. Impute the Fence:
# this variable represents the Fence quality. Like the other variables which have lots of missing values
# When "Fence" equals to 0, it means that there is no fence.
# So I decided to impute this feature by "None" 

print(pd.Series(all_data["Fence"]).value_counts())
all_data["Fence"] = all_data["Fence"].fillna("None")
# 5. Impute the FireplaceQu:
# this variable represents the Number of fireplaces.Comparing the variable "Fireplaces", 
# it could be easy to find that the NA value means there is no value for fireplaces.
# When "Fireplaces" equals to 0, the value of "FireplaceQu" is 0. So I decided to impute this feature by "None" 

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
# 6. the other categorical variables which have many missing values:
## "GarageYrBlt","GarageType", "GarageFinish", "GarageQual", "GarageCond","BsmtFinType2", "BsmtExposure", "BsmtFinType1", 
#"BsmtCond", "BsmtQual"
# If the values are missing, it means there is no garage or basement for the house. So impute "None".

for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond", 
            "BsmtFinType1", "BsmtFinType2", "BsmtFinType", "BsmtExposure", "BsmtCond","BsmtQual"]:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')
# 7. The other categorical variables which have few missing values:
# so impute these variables with the most common element


all_data["Electrical"] = all_data["Electrical"].fillna(all_data['Electrical'].mode()[0])


all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data['KitchenQual'].mode()[0] )
all_data["Exterior1st"] = all_data["Exterior1st"].fillna(all_data['Exterior1st'].mode()[0] )
all_data["Exterior2nd"] = all_data["Exterior2nd"].fillna(all_data['Exterior2nd'].mode()[0] )
all_data["SaleType"] = all_data["SaleType"].fillna(all_data['SaleType'].mode()[0] )
all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data['MSZoning'].mode()[0] )


all_data["Functional"] = all_data["Functional"].fillna(all_data['Functional'].mode()[0] )
all_data["Utilities"] = all_data["Utilities"].fillna(all_data['Utilities'].mode()[0] )

# Impute numerical variables:
## 8. Impute the numerical data
# ["LotFrontage", "Electrical"]
# For logFrontage: Linear feet of street connected to property
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
## 9.  Numerical variables : "GarageYrBlt", "GarageArea", 'GarageCars", "GarageYrBlt"

for col in ("GarageYrBlt", "GarageArea", "GarageCars"):
    all_data[col] = all_data[col].fillna(0)
# 10. the numerical variables for the basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
# 12. For "MasVnrArea" and "MasVnrType":
# when "MasVnrType" = none, "MasVnrArea" = 0. Hence, impute "MasVnrType" with "none" and impute "MasVnrArea" with zero.

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
# Last chack if there is any missing value left
missing_val_col_check= (all_data.isnull().sum())
missing_val_col_percent_check = missing_val_col_check / all_data.shape[0]
missing_percent_check = missing_val_col_percent_check[ missing_val_col_percent_check > 0 ].sort_values(ascending=False)
missing_percent_check

# PoolQC          0.996574
# MiscFeature     0.964029
# Alley           0.932169
# Fence           0.804385
# FireplaceQu     0.486468

# all_data.drop(["PoolQC"], axis=1, inplace=True)

# all_data.drop(["MiscFeature"], axis=1, inplace=True)
# all_data.drop(["Alley"], axis=1, inplace=True)
# all_data.drop(["Fence"], axis=1, inplace=True)

# all_data.drop(["FireplaceQu"], axis=1, inplace=True)

n_train = dt_train.shape[0]
n_test = dt_test.shape[0]

n_train
all_data.shape
## Before this part, I transform "numerical" data as categorical data. 
## Some variables like the year and month which have few types:
##     "YearBuilt", "YearRemodAdd", "GarageYrBlt", "MoSold", "YrSold"
## num:    112           61             98              12      5

# Hence, convert "MoSold", "YrSold" to categorical data
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
# Some other variables like  MSSubClass WHICH indicates the class of the building.
# Only 16 types of this variable.

pd.Series(all_data["MSSubClass"]).value_counts()
all_data["MSSubClass"] = all_data["MSSubClass"].apply(str)
# and variable OverallCond which indicates the categorical variable 
# Here are only 9 types of this variable. It is better to convert it to categorical
pd.Series(all_data["OverallCond"]).value_counts()

all_data["OverallCond"] = all_data["OverallCond"].apply(str)
print(all_data.shape)
## Version1: donot delete any variables
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
## Delete 5 variables  missing > 80%
from sklearn.preprocessing import LabelEncoder
cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC',  'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street',  'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
# Adding total sqfootage feature 极有可能 增加 noise  通过对比
#    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data_numerical = all_data.dtypes[all_data.dtypes != "object"].index
all_data_numerical

len(all_data_numerical)
# The skew of all numerical features
skewed_feats = all_data[all_data_numerical].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness

# The skew of all numerical features
skewed_feats = all_data[all_data_numerical].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness

##  For very high skewness
skewness_h = skewness[abs(skewness) > 20] 

lam_h = 0.1
for feat in skewness_h.index:
    all_data[feat] = boxcox1p(all_data[feat], lam_h)

## When the value of skewness is larger than 1, the distribution of variable is highly skewed
## re-determine the value to transform at 0.7
skewness1 = skewness[abs(skewness) > 0.5] 
print("There are {} skewed numerical features to Box Cox transform".format(skewness1.shape[0]))

highly_skewed = skewness1.index
lam = 0.15
for feat in highly_skewed:
    all_data[feat] = boxcox1p(all_data[feat], lam)

target_train = dt_train["SalePrice"]
target_train.head()
## From the EDA part, it can be easily found that the target value "SalePrice" is highly skewed.
# So using boxcox transformation for the target value as well.
transformed_target = boxcox1p( target_train, lam)

sns.distplot( transformed_target )



##transformed_target2 = np.log( target_train )
##sns.distplot( transformed_target2 )
skew(transformed_target)
all_data.shape
all_data = pd.get_dummies(all_data)
print(all_data.shape)

all_data.head(5)


## delete 5 variables.   //  add 2 more variables which are transfromed from numerical to categorical
## 3. XGboost
### Improvements
xgb_model = XGBRegressor()
# 减小 gamma,  提高 max_depth,

parameters_xgb = {"min_child_weight":[3,4,5], "gamma":[0.6, 0.5, 0.3, 0.1,0.01], 
                  "subsample":[1, 0.8, 0.6,0.4],"colsample_bytree":[i/10.0 for i in range(5,11)],
                  "max_depth": [4,5,6], "random_state": [1,2,5,6]}

grid_xgb = GridSearchCV(xgb_model, parameters_xgb)

grid_xgb.fit(x_train, y_train)
xgb_best = grid_xgb.best_estimator_
xgb_best.fit(x_train, y_train)
print("the information of Best RandomForestRegressor Model:\n\n " + str(grid_xgb.best_estimator_) + "\n")
print("Best Score: " + str(grid_xgb.best_score_) + "\n")
## 最终 xgg 的参数：
#  base_score=0.5, booster='gbtree', colsample_bylevel=1,      colsample_bytree=1.0, gamma=0.3, learning_rate=0.1,
#   max_delta_step=0, max_depth=5, min_child_weight=4, missing=None,  n_estimators=100, n_jobs=1, nthread=None, 
#  objective='reg:linear',     random_state=5, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#  seed=None, silent=True, subsample=0.8
## Improvements
print("the information of Best RandomForestRegressor Model:\n\n " + str(grid_xgb.best_estimator_) + "\n")
print("Best Score: " + str(grid_xgb.best_score_) + "\n")

xgb_pred = xgb_best.predict(x_validation)
r2_xgb = r2_score(y_validation, xgb_pred)

rmse_xgb = np.sqrt(mean_squared_error(y_validation, xgb_pred))
print("test R^2 Score: " + str(r2_xgb))
print("test RMSE Score: " + str(rmse_xgb))


scores_xgb = cross_val_score(xgb_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_xgb)))
## 3. XGboost
### Improvements
xgb_model = XGBRegressor()
# 减小 gamma,  提高 max_depth,

parameters_xgb = {"min_child_weight":[3,4,5], "gamma":[0.6, 0.5, 0.3, 0.1,0.01], 
                  "subsample":[1, 0.8, 0.6,0.4],"colsample_bytree":[i/10.0 for i in range(5,11)],
                  "max_depth": [4,5,6], "random_state": [1,2,5,6]}

grid_xgb = GridSearchCV(xgb_model, parameters_xgb)

grid_xgb.fit(x_train, y_train)
xgb_best = grid_xgb.best_estimator_
xgb_best.fit(x_train, y_train)
print("the information of Best RandomForestRegressor Model:\n\n " + str(grid_xgb.best_estimator_) + "\n")
print("Best Score: " + str(grid_xgb.best_score_) + "\n")
## Improvements
print("the information of Best RandomForestRegressor Model:\n\n " + str(grid_xgb.best_estimator_) + "\n")
print("Best Score: " + str(grid_xgb.best_score_) + "\n")

xgb_pred = xgb_best.predict(x_validation)
r2_xgb = r2_score(y_validation, xgb_pred)

rmse_xgb = np.sqrt(mean_squared_error(y_validation, xgb_pred))
print("test R^2 Score: " + str(r2_xgb))
print("test RMSE Score: " + str(rmse_xgb))


scores_xgb = cross_val_score(xgb_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_xgb)))
##  1. lasso
# Using GridSearch CV to find the best 

lasso = Lasso()
grid_lasso = GridSearchCV(lasso, {"alpha": [0.1, 0.01,0.001, 0.0001],"fit_intercept" : [True, False], 
                                  "normalize" : [True, False],"precompute" : [True, False], "max_iter" :[10000], 
                                  "copy_X" : [True, False]},verbose=1, scoring="r2", cv = 5)
grid_lasso.fit(x_train, y_train)

lasso_best = grid_lasso.best_estimator_
lasso_best.fit(x_train, y_train)
print("the information of Best Lasso Model: \n\n" + str(grid_lasso.best_estimator_) + "\n")
print("Best Score: " + str(grid_lasso.best_score_)  + "\n" )

print("the information of Best Lasso Model: \n\n" + str(grid_lasso.best_estimator_) + "\n")
print("Best Score: " + str(grid_lasso.best_score_)  + "\n" )

lasso_pred = lasso_best.predict(x_validation)
r2_lasso = r2_score(y_validation, lasso_pred)

rmse_lasso = np.sqrt(mean_squared_error(y_validation, lasso_pred))

print("test R^2 Score: " + str(r2_lasso))
print("test RMSE Score: " + str(rmse_lasso))

scores_lasso = cross_val_score(lasso_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lasso)))


##  1. lasso
# Using GridSearch CV to find the best 

lasso = Lasso()
grid_lasso = GridSearchCV(lasso, {"alpha": [0.1,0.01,0.001,0.0001],"fit_intercept" : [True, False], 
                                  "normalize" : [True, False],"precompute" : [True, False], "max_iter" :[10000, 50000], 
                                  "copy_X" : [True, False]},verbose=1, scoring="r2", cv = 5)
grid_lasso.fit(x_train, y_train)

lasso_best = grid_lasso.best_estimator_
lasso_best.fit(x_train, y_train)
print("the information of Best Lasso Model: \n\n" + str(grid_lasso.best_estimator_) + "\n")
print("Best Score: " + str(grid_lasso.best_score_)  + "\n" )

print("the information of Best Lasso Model: \n\n" + str(grid_lasso.best_estimator_) + "\n")
print("Best Score: " + str(grid_lasso.best_score_)  + "\n" )

lasso_pred = lasso_best.predict(x_validation)
r2_lasso = r2_score(y_validation, lasso_pred)

rmse_lasso = np.sqrt(mean_squared_error(y_validation, lasso_pred))

print("test R^2 Score: " + str(r2_lasso))
print("test RMSE Score: " + str(rmse_lasso))

scores_lasso = cross_val_score(lasso_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lasso)))

## 弄一个新版本  submission
## Get the prediction results by the lasso algorithm
from scipy.special import boxcox, inv_boxcox
x_test = test
submission_predictions = inv_boxcox(lasso_best.predict(x_test), 0.15)
submission_data2 = pd.DataFrame({"Id" : dt_test["Id"], "SalePrice": submission_predictions })
submission_data2.to_csv("lasso 4th No deleting.csv")

##  5. lasso
# Using GridSearch CV to find the best 

lasso = Lasso()
grid_lasso = GridSearchCV(lasso, {"alpha": [0.1,0.01,0.001,0.0001],"fit_intercept" : [True, False], 
                                  "normalize" : [True, False],"precompute" : [True, False], "max_iter" :[10000,50000], 
                                  "copy_X" : [True, False]},verbose=1, scoring="r2", cv = 5)
grid_lasso.fit(x_train, y_train)

lasso_best = grid_lasso.best_estimator_
lasso_best.fit(x_train, y_train)
print("the information of Best Lasso Model: \n\n" + str(grid_lasso.best_estimator_) + "\n")
print("Best Score: " + str(grid_lasso.best_score_)  + "\n" )

lasso_pred = lasso_best.predict(x_validation)
r2_lasso = r2_score(y_validation, lasso_pred)

rmse_lasso = np.sqrt(mean_squared_error(y_validation, lasso_pred))

print("test R^2 Score: " + str(r2_lasso))
print("test RMSE Score: " + str(rmse_lasso))

scores_lasso = cross_val_score(lasso_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lasso)))



##  1. lasso
# Using GridSearch CV to find the best 

lasso = Lasso()
grid_lasso = GridSearchCV(lasso, {"alpha": [0.1,0.01,0.001,0.0001],"fit_intercept" : [True, False], 
                                  "normalize" : [True, False],"precompute" : [True, False], "max_iter" :[10000], 
                                  "copy_X" : [True, False]},verbose=1, scoring="r2", cv = 5)
grid_lasso.fit(x_train, y_train)

lasso_best = grid_lasso.best_estimator_
lasso_best.fit(x_train, y_train)
print("the information of Best Lasso Model: \n\n" + str(grid_lasso.best_estimator_) + "\n")
print("Best Score: " + str(grid_lasso.best_score_)  + "\n" )

lasso_pred = lasso_best.predict(x_validation)
r2_lasso = r2_score(y_validation, lasso_pred)

rmse_lasso = np.sqrt(mean_squared_error(y_validation, lasso_pred))

print("test R^2 Score: " + str(r2_lasso))
print("test RMSE Score: " + str(rmse_lasso))

scores_lasso = cross_val_score(lasso_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lasso)))

from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

import xgboost as xgb
from xgboost import XGBRegressor


from mlxtend.regressor import StackingRegressor

##
train = all_data[:n_train]
test = all_data[n_train:]
train.shape
x_train, x_validation ,y_train, y_validation = train_test_split(train, transformed_target, test_size = 0.2)
print(x_train.shape)
print(y_train.shape)
lasso.get_params()

##  1. lasso
# Using GridSearch CV to find the best 

lasso = Lasso()
grid_lasso = GridSearchCV(lasso, {"alpha": [0.1,0.01,0.001,0.0001],"fit_intercept" : [True, False], 
                                  "normalize" : [True, False],"precompute" : [True, False], "max_iter" :[10000], 
                                  "copy_X" : [True, False]},verbose=1, scoring="r2", cv = 5)
grid_lasso.fit(x_train, y_train)

lasso_best = grid_lasso.best_estimator_
lasso_best.fit(x_train, y_train)
print("the information of Best Lasso Model: \n\n" + str(grid_lasso.best_estimator_) + "\n")
print("Best Score: " + str(grid_lasso.best_score_)  + "\n" )

lasso_pred = lasso_best.predict(x_validation)
r2_lasso = r2_score(y_validation, lasso_pred)

rmse_lasso = np.sqrt(mean_squared_error(y_validation, lasso_pred))

print("test R^2 Score: " + str(r2_lasso))
print("test RMSE Score: " + str(rmse_lasso))

scores_lasso = cross_val_score(lasso_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lasso)))

## 2. Random Forest Regression
#
rf = RandomForestRegressor()
paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 
                 "max_features" : ["auto", "log2"]}
grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2", cv = 5)
grid_rf.fit(x_train, y_train)

print("the information of Best RandomForestRegressor Model:\n\n " + str(grid_rf.best_estimator_) + "\n")
print("Best Score: " + str(grid_rf.best_score_) + "\n" )

rf_best = grid_rf.best_estimator_

rf_best.fit(x_train, y_train)
rf_pred = rf_best.predict(x_validation)
r2_rf = r2_score(y_validation, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_validation, rf_pred))
print("test R^2 Score: " + str(r2_rf))
print("test RMSE Score: " + str(rmse_rf))


scores_rf = cross_val_score(rf_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_rf)))

## 3. XGboost
xgb_model = XGBRegressor()


parameters_xgb = {"min_child_weight":[3,4,5], "gamma":[i/10.0 for i in range(3,6)], 
                  "subsample":[i/10.0 for i in range(6,11)],"colsample_bytree":[i/10.0 for i in range(5,11)],
                  "max_depth": [3,4,5,6]}

grid_xgb = GridSearchCV(xgb_model, parameters_xgb)

grid_xgb.fit(x_train, y_train)
xgb_best = grid_xgb.best_estimator_
xgb_best.fit(x_train, y_train)

## 3. XGboost
### Improvements
xgb_model = XGBRegressor()
# 减小 gamma,  提高 max_depth,

parameters_xgb = {"min_child_weight":[3,4,5], "gamma":[0.6, 0.5, 0.3, 0.1, 0.05,0.01], 
                  "subsample":[1, 0.8, 0.6,0.4,0.1],"colsample_bytree":[i/10.0 for i in range(5,11)],
                  "max_depth": [4,5,6,8], "random_state": [0,1,2,5]}

grid_xgb = GridSearchCV(xgb_model, parameters_xgb)

grid_xgb.fit(x_train, y_train)
xgb_best = grid_xgb.best_estimator_
xgb_best.fit(x_train, y_train)
print("the information of Best RandomForestRegressor Model:\n\n " + str(grid_xgb.best_estimator_) + "\n")
print("Best Score: " + str(grid_xgb.best_score_) + "\n")
## 最终 xgg 的参数：
#  base_score=0.5, booster='gbtree', colsample_bylevel=1,      colsample_bytree=1.0, gamma=0.3, learning_rate=0.1,
#   max_delta_step=0, max_depth=5, min_child_weight=4, missing=None,  n_estimators=100, n_jobs=1, nthread=None, 
#  objective='reg:linear',     random_state=5, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#  seed=None, silent=True, subsample=0.8
## Improvements
print("the information of Best RandomForestRegressor Model:\n\n " + str(grid_xgb.best_estimator_) + "\n")
print("Best Score: " + str(grid_xgb.best_score_) + "\n")

xgb_pred = xgb_best.predict(x_validation)
r2_xgb = r2_score(y_validation, xgb_pred)

rmse_xgb = np.sqrt(mean_squared_error(y_validation, xgb_pred))
print("test R^2 Score: " + str(r2_xgb))
print("test RMSE Score: " + str(rmse_xgb))


scores_xgb = cross_val_score(xgb_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_xgb)))
print("the information of Best RandomForestRegressor Model:\n\n " + str(grid_xgb.best_estimator_) + "\n")
print("Best Score: " + str(grid_xgb.best_score_) + "\n")

xgb_pred = xgb_best.predict(x_validation)
r2_xgb = r2_score(y_validation, xgb_pred)

rmse_xgb = np.sqrt(mean_squared_error(y_validation, xgb_pred))
print("test R^2 Score: " + str(r2_xgb))
print("test RMSE Score: " + str(rmse_xgb))

# reg_alpha reg_lambda

scores_xgb = cross_val_score(xgb_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_xgb)))

## 4. Stacking Approach
ridge = Ridge()
lasso = Lasso()
rf = RandomForestRegressor()
xgb_model = XGBRegressor()

svr= SVR(kernel= 'rbf', C = 20)


stacking = StackingRegressor(regressors= [lasso_best, xgb_best], meta_regressor= svr)
params = {'lasso__alpha': [0.1, 1.0, 10.0]}

grid_stacking = GridSearchCV(stacking, params, cv=5, verbose=1, refit=True)

grid_stacking.fit(x_train, y_train)

stacking_best = grid_stacking.best_estimator_
stacking_best.fit(x_train, y_train)


## 4. Stacking Approach

## Inprovments
ridge = Ridge()
lasso = Lasso()
rf = RandomForestRegressor()
xgb_model = XGBRegressor()

svr= SVR(kernel= 'rbf', C = 20)


stacking = StackingRegressor(regressors= [rf_best, xgb_best], meta_regressor= lasso_best)
#params = {'lasso__alpha': [0.1, 1.0, 10.0]}
params_stacking = {'meta-lasso__alpha': [ 0.1, 0.01,0.001, 0.005 ,0.0001]}

grid_stacking = GridSearchCV(stacking, params_stacking, cv=5, verbose=1, refit=True)

grid_stacking.fit(x_train, y_train)

stacking_best = grid_stacking.best_estimator_
stacking_best.fit(x_train, y_train)


## Inprovements
print("the information of Best stacking Model: \n\n" + str(grid_stacking.best_estimator_) + "\n")
print("Best Score: " + str(grid_stacking.best_score_)  + "\n")

stacking_pred = stacking_best.predict(x_validation)
r2_stacking = r2_score(y_validation, stacking_pred)

rmse_stacking = np.sqrt(mean_squared_error(y_validation, stacking_pred))

print("test R^2 Score: " + str(r2_stacking))
print("test RMSE Score: " + str(rmse_stacking))

scores_stacking = cross_val_score(stacking_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_stacking)))

print("the information of Best stacking Model: \n\n" + str(grid.best_estimator_) + "\n")
print("Best Score: " + str(grid.best_score_)  + "\n")

stacking_pred = stacking_best.predict(x_validation)
r2_stacking = r2_score(y_validation, stacking_pred)

rmse_stacking = np.sqrt(mean_squared_error(y_validation, stacking_pred))

print("test R^2 Score: " + str(r2_stacking))
print("test RMSE Score: " + str(rmse_stacking))

scores_stacking = cross_val_score(stacking_best, x_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_stacking)))


# The table of results for 4 different algorithm
results = pd.DataFrame({
    "Model" : [ "Lasso", "Random Forest Regressor"," XGboost", "Stacking Approach" ],
    "Best Score" : [grid_lasso.best_score_,  grid_rf.best_score_, grid_xgb.best_score_, grid_stacking.best_score_],
    "Test R^2 Score" : [str(r2_lasso)[0:5], str(r2_rf)[0:5],  str(r2_xgb)[0:5], str(r2_stacking)[0:5]],
    "RMSE" : [str(rmse_lasso)[0:8], str(rmse_rf)[0:8],  str(rmse_xgb)[0:8], str(rmse_stacking)[0:8]]
})
results
test.shape
test.head(5)
## Get the prediction results by the lasso algorithm
from scipy.special import boxcox, inv_boxcox
x_test = test
submission_predictions = inv_boxcox(lasso_best.predict(x_test), 0.15)
sum_predictions2 = inv_boxcox(xgb_best.predict(x_test), 0.15)
sum_predictions2
submission_data2 = pd.DataFrame({"Id" : dt_test["Id"], "SalePrice": sum_predictions2 })
submission_data2.to_csv("submission2.csv")
submission3 = inv_boxcox(stacking_best.predict(x_test), 0.15)

submission_data3 = pd.DataFrame({"Id" : dt_test["Id"], "SalePrice": submission3 })
submission_data3.to_csv("submission3.csv")
submission_data = pd.DataFrame({"Id" : dt_test["Id"], "SalePrice": submission_predictions })
submission_data

submission_data.to_csv("submission_data.csv")



