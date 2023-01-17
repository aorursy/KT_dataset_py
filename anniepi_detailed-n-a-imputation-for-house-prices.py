# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import skew
from IPython.display import display
%matplotlib inline

pd.set_option("max_columns", None)
# Load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Check for duplicates and sizes of the training and testing set
print("Training")
print(train.shape)
print(len(train.Id.unique()))
print("\n")
print("Testing")
print(test.shape)
print(len(test.Id.unique()))
# Plot histogram of SalePrice and log(SalePrice)
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
sns.distplot(train.SalePrice)
plt.subplot(1,2,2)
sns.distplot(np.log(train.SalePrice))
train['SalePrice'] = np.log1p(train['SalePrice'])
# Create Saleprice correlation matrix
k = 10 #number of variables for heatmap
corrmat = train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# Plot grlivarea vs. saleprice and overallqual vs. saleprice
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(train.OverallQual, train.SalePrice, marker = "s")
plt.title("OverallQual vs. SalePrice")
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")

plt.subplot(1,2,2)
plt.scatter(train.GrLivArea, train.SalePrice, marker = "s")
plt.title("GrLivArea vs. SalePrice")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
# Drop two outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<12.5)].index)
# Save ID values
train_ID = train['Id']
test_ID = test['Id']

# Save shapes
ntrain = train.shape[0]
ntest = test.shape[0]

# Store Sale Price as target value
y_target = train.SalePrice.values

# Combine train and test into new dataset called Combined and drop SalePrice
combined = pd.concat((train, test)).reset_index(drop=True)
combined.drop(['SalePrice'], axis=1, inplace=True)
print("combined size is : {}".format(combined.shape))
# Select only numerical variables
numerical_features = combined.select_dtypes(exclude = ["object"])

# Inspect summary statistics for each numerical variable
numerical_features.describe()
combined[combined["GarageYrBlt"] == 2207]
combined.loc[combined["GarageYrBlt"] == 2207,'GarageYrBlt'] = 2007 
# NA discovery
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")
# Create function to detect rows with missing values greater than 0, but less than number of variables being passed
def countMissing(var):
    print(var)
    for index, row in combined.iterrows():
        missing = 0
        for i in var:
            if (type(row[i]) == float) or (type(row[i]) == int):
                if (row[i] == 0):
                    missing += 1 
                else:
                    if (np.isnan(row[i])) or (row[i] == "None"):
                        missing += 1
            else:
                if (row[i] == "None"):
                    missing += 1
        if (missing > 0) and (missing < len(var)):
            print("Index: " + str(index) + " Number Missing: " + str(missing))
    print("\n")
# Create lists of related variables
poolVariables = ["PoolQC", "PoolArea"]
basementVariables = ["BsmtQual", "BsmtCond", "BsmtExposure", "TotalBsmtSF"]
garageVariables = ["GarageType", "GarageYrBlt", "GarageArea"]
veneerVariables = ["MasVnrArea", "MasVnrType"]
roofVariables = ["RoofStyle", "RoofMatl"]
kitchenVariables = ["KitchenAbvGr", "KitchenQual"]
fireplaceVariables = ["Fireplaces", "FireplaceQu"]

# Call function on all lists of related variables
for i in (poolVariables, basementVariables, garageVariables, veneerVariables, roofVariables, kitchenVariables, fireplaceVariables):
    countMissing(i)
# Inspect rows flagged for missing Pool variables
combined.loc[[2418,2501,2597]]
# Inspect rows with PoolArea and PoolQC filled in
combined[(combined["PoolArea"].notnull()) & (combined["PoolArea"] != 0)]
# Impute missing values for PoolQC
combined.loc[2418, 'PoolQC'] = 'Fa' 
combined.loc[2501, 'PoolQC'] = 'Gd' 
combined.loc[2597, 'PoolQC'] = 'Fa'
# Inspect rows flagged for missing Basement variables
combined.loc[[947,1485,2038,2183,2215,2216,2346,2522]]
# Inspect values for BsmtCond, BsmtExposure, and BsmtQual
print(combined['BsmtCond'].value_counts())
print("\n")
print(combined['BsmtQual'].value_counts())
print("\n")
print(combined['BsmtExposure'].value_counts())
# Impute missing Basement values
combined.loc[947, 'BsmtExposure'] = 'No' 
combined.loc[1485, 'BsmtExposure'] = 'No' 
combined.loc[2346, 'BsmtExposure'] = 'No'
combined.loc[2038, 'BsmtCond'] = 'TA'
combined.loc[2183, 'BsmtCond'] = 'TA'
combined.loc[2346, 'BsmtCond'] = 'TA'
combined.loc[2215, 'BsmtQual'] = 'TA'
combined.loc[2216, 'BsmtQual'] = 'TA'
# Inspect rows flagged for missing Veneer variables
combined.loc[[623,687,772,1229,1240,1298,1332,1667,2317,2450,2608]]
# Show summary statistics for MasVnrArea by MasVnrType
combined.groupby(['MasVnrType'])['MasVnrArea'].describe()
# Fill in rows with MasVnrArea but no MasVnrType
for i in (623,2608,1298,1332,1667):
    combined.loc[i, "MasVnrType"] = "BrkFace"

# Fill in rows with no MasVnrArea but a MasVnrType
combined.loc[687, 'MasVnrArea'] = 261.67
combined.loc[1240, 'MasVnrArea'] = 247
combined.loc[2317, 'MasVnrArea'] = 261.67

#Fill in rows with 1 for MasVnr area and None for MAsVnrType
for i in (772,1249,2450):
    combined.loc[i, "MasVnrArea"] = 0
# Inspect rows flagged for missing Garage variables
combined.loc[[2124,2574]]
# Impute missing Garage values
combined.loc[2124, 'GarageYrBlt'] = combined.loc[2124, 'YearBuilt']
combined.loc[2124, 'GarageCond'] = combined['GarageCond'].mode()[0]
combined.loc[2124, 'GarageFinish'] = combined['GarageFinish'].mode()[0]
combined.loc[2124, 'GarageQual'] = combined['GarageQual'].mode()[0]

combined.loc[2574, 'GarageType'] = 'None' 
# Inspect rows flagged for missing Kitchen values
combined.loc[[953,1553,2585,2857]]
# Inspect value counts for KitchenAbvGr and KitchenQual
print(combined["KitchenAbvGr"].value_counts())
print("\n")
print(combined["KitchenQual"].value_counts())
# Impute missing KitchenAbvGr values
for i in (953,2585,2857):
    combined.loc[i, "KitchenAbvGr"] = 1

# Impute missing KitchenQual values
combined.loc[1553, "KitchenQual"] = "TA"
# Impute missing values based on Data Description

# Alley: Data description says NA means 'no alley access'
combined["Alley"].fillna("None", inplace=True)

# Bsmt : Data description says NA for basement features is "no basement"
combined["BsmtQual"].fillna("None", inplace=True)
combined["BsmtCond"].fillna("None", inplace=True)
combined["BsmtExposure"].fillna("None", inplace=True)
combined["BsmtFinType1"].fillna("None", inplace=True)
combined["BsmtFinType2"].fillna("None", inplace=True)

# Fence : Data description says NA means "no fence"
combined["Fence"].fillna("None", inplace=True)

# FireplaceQu : Data description says NA means "no fireplace"
combined["FireplaceQu"].fillna("None", inplace=True)

# Garage : Data description says NA for garage features is "no garage"
combined["GarageType"].fillna("None", inplace=True)
combined["GarageFinish"].fillna("None", inplace=True)
combined["GarageQual"].fillna("None", inplace=True)
combined["GarageCond"].fillna("None", inplace=True)

# MiscFeature : Data description says NA means "no misc feature"
combined["MiscFeature"].fillna("None", inplace=True)

# PoolQC : Data description says NA means "no pool"
combined["PoolQC"].fillna("None", inplace=True)

# Print remaining variables with missing values
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")
# Check rows where BsmtFullBath is null
combined[combined.BsmtFullBath.isnull()]
# Impute missing values for Basement and Garage variables
combined["BsmtFullBath"].fillna(0, inplace=True)
combined["BsmtHalfBath"].fillna(0, inplace=True)
combined["BsmtFinSF1"].fillna(0, inplace=True)
combined["BsmtFinSF2"].fillna(0, inplace=True)
combined["BsmtUnfSF"].fillna(0, inplace=True)
combined["TotalBsmtSF"].fillna(0, inplace=True)
combined["GarageArea"].fillna(0, inplace=True)
combined["GarageCars"].fillna(0, inplace=True)

# Print remaining variables with missing values
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")
# Check most common MSZoning values for each MSSubClass
combined.groupby(['MSSubClass'])['MSZoning'].describe()
# Inspect missing MSZoning values
combined[combined["MSZoning"].isnull()]
# Impute missing MSZoning values based on mode by MSSubClass
combined.loc[1913,"MSZoning"] = "RM"
combined.loc[2214,"MSZoning"] = "RL"
combined.loc[2248,"MSZoning"] = "RM"
combined.loc[2902,"MSZoning"] = "RL"
# Check value counts for variables with only 1 or 2 missing variables
for col in ('Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'SaleType', 'Utilities', 'KitchenQual'):
    print(combined[col].value_counts())
    print("\n")
# Drop utilities
combined = combined.drop(['Utilities'], axis=1)

# Fill in mode for other variables
for col in ('Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'SaleType', 'KitchenQual'):
    combined[col].fillna(combined[col].mode()[0], inplace=True)
    
# Print remaining variables with missing values
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")
combined[combined["GarageYrBlt"].isnull()]
# Impute missing values for GarageYrBuilt
combined["GarageYrBlt"].fillna(0, inplace=True)
# Plot distribution of LotFrontage
sns.distplot(combined['LotFrontage'].dropna());
# Show summary statistics for LotFrontage by Neighborhood
combined.groupby("Neighborhood")['LotFrontage'].describe()
# Impute median LotFrontage by neighborhood for missing values
combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
# Impute missing values for MasVnrArea and MasVnrType
combined["MasVnrArea"].fillna(0, inplace=True)
combined["MasVnrType"].fillna("None", inplace=True)
# Print remaining variables iwth missing columns
naCols = combined.isnull().sum()[combined.isnull().sum()>0]
print(naCols)
print("\n")
print("There are " + str(len(naCols)) + " columns with missing values.")
# Replace values for MSSubclass and MoSold
combined = combined.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })
skew = combined.skew(numeric_only=True) > 0.75

print(skew[skew==True])
print("\n")
print("There are " + str(len(skew[skew==True])) + " skewed variables.")
# Apply log transformation to skew values in training
from scipy.special import boxcox1p

for index_val, series_val in skew.iteritems():
    if series_val == True:
        combined[index_val] = boxcox1p(combined[index_val], 0.15)
        
skew = combined.skew(numeric_only=True) > 0.75

# Print remaining skew variables
print(skew[skew==True])
print("\n")
print("There are " + str(len(skew[skew==True])) + " skewed variables.")
plt.figure(figsize=(20,10))

# Plot histogram for remaining skew variables
for col in ('3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'EnclosedPorch', 'KitchenAbvGr', 'LowQualFinSF', 'MiscVal', 'PoolArea', 'ScreenPorch'):
    plt.figure()
    sns.distplot(combined[col])
combined["YearsSinceRemodelled"] = combined["YrSold"] - combined["YearRemodAdd"]
# Check if YearsSinceRemodelled is less than 0
combined[combined["YearsSinceRemodelled"] < 0]
# Update YrSold for two rows
combined.loc[2293, 'YrSold'] = 2008
combined.loc[2547, 'YrSold'] = 2009
# Create new variable: total square footage
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
# Concatenate categorical and numerical features
print("Previous number of variables: " + str(len(combined.columns)))
combined = pd.get_dummies(combined)
print("New number of variables: " + str(len(combined.columns)))
# Create function to check for variables with near zero variance (99.9% zero values)
def countZeroes(var):
    nearZeroVariables = []
    for i in var:
        zeroValues = 0
        for index, row in combined.iterrows():
            if row[i] == 0:
                zeroValues += 1
        if zeroValues > 0.999 * len(combined):
            nearZeroVariables.append(i)
            print("Variable " + str(i) + ": " + str(zeroValues))
    combined.drop(nearZeroVariables, axis=1, inplace=True)
# Run function on all columns and drop ones with 99.9% zero values
colnames = list(combined)
countZeroes(colnames)
# Drop additional variables to prevent overfitting
combined.drop(["MSZoning_C (all)", "Condition2_PosN", "MSSubClass_SC160", "Street_Grvl", "Street_Pave"], axis=1, inplace=True)
# Display how many columns are remaining
len(combined.columns)
# Split into train and test again and delete ID
train = combined[:ntrain]
test = combined[ntrain:]

print(train.shape)
print(test.shape)

del train["Id"]
del test["Id"]
# Check for outliers
import statsmodels.api as sm

ols = sm.OLS(endog = y_target, exog = train)
fit = ols.fit()
otest = fit.outlier_test()['bonf(p)']

outliers = list(otest[otest<1e-3].index) 

outliers
# Drop outliers from train
for index in sorted(outliers, reverse=True):
    train = train.drop([index])
# Drop outliers from y_target (SalesPrice)
y_target = np.delete(y_target, outliers)
# Import libraries for modeling
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from vecstack import stacking
# Local validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_target, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
# Lasso model
lasso = make_pipeline(RobustScaler(), LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Ridge model
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]))
score = rmsle_cv(ridge)
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Elastic net model
ENet = make_pipeline(RobustScaler(), ElasticNetCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Fit lasso to train and y_target
lasso2 = lasso.fit(train, y_target)
# Plot feature importance in lasso model
coef = pd.Series(lasso2.steps[1][1].coef_, index = train.columns)

imp_coef = pd.concat([coef.sort_values().head(15),
                     coef.sort_values().tail(15)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
# Plotting feature importance in Ridge Model

ridge2 = ridge.fit(train, y_target)
coef = pd.Series(ridge2.steps[1][1].coef_, index = train.columns)

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")
# Plotting feature importance in Elastic Model

ENet2 = ENet.fit(train, y_target)
coef = pd.Series(ENet2.steps[1][1].coef_, index = train.columns)

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Elastic Net Model")
# Select models for stacking
models = [lasso, ridge, ENet]
    
# Compute stacking features
S_train, S_test = stacking(models, train, y_target, test, 
    regression = True, metric = mean_absolute_error, n_folds = 4, 
    shuffle = True, random_state = 0, verbose = 2)

# Initialize 2nd level model
model = make_pipeline(RobustScaler(), ElasticNetCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], l1_ratio=.9, random_state=3))
    
# Fit 2nd level model
model = model.fit(S_train, y_target)

# Predict
y_pred = np.expm1(model.predict(S_test))
# Test stacked model RSME
from sklearn.model_selection import cross_val_predict

score_preds = cross_val_predict(model, X=S_train, y=y_target)
print("stacked RMSE = ", np.sqrt(np.mean((y_target - score_preds)**2)))
# Check my Sales Price predictions before submitting
y_pred
# Create submission
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = y_pred
submission.to_csv('submission_v16.csv',index=False)