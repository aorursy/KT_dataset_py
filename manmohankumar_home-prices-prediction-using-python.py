# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from scipy.stats import norm

from scipy.stats import skew

from scipy.stats.stats import pearsonr

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

%matplotlib inline

warnings.filterwarnings('ignore')



#Loading the data

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.shape
sns.distplot(train_data['SalePrice'])

plt.title("Distribution of Sale Price")
train_data['SalePrice'].describe()
train_data['SalePrice'].median()
"SalePrice Skewness measure: {0}, and Kurtosis measure is: {1}".format(train_data['SalePrice'].skew(), train_data['SalePrice'].kurt())
# Looking at correlation with saleprice

corr_mat = train_data.corr()

#corr_mat['SalePrice']

corr_mat.nlargest(10, 'SalePrice')['SalePrice']
# looking at saleprice correlation matrix for top 10 closely related variables

corr_mat = train_data.corr()

cols = corr_mat.nlargest(10, 'SalePrice')['SalePrice'].index

correlation_matrix = np.corrcoef(train_data[cols].values.T)

plt.figure(figsize=(9,9))

sns.heatmap(correlation_matrix, vmax=.8, square=True, annot=True, yticklabels=cols.values,

            xticklabels=cols.values, fmt='.2f',  annot_kws={'size': 10}, cmap='viridis',

           linewidths=0.01, linecolor='white')

plt.title('Correlation between top 10 related features')

plt.show()
# Checking for count of missing Data in features, in descending order of count

missing_data = train_data.columns[train_data.isnull().any()]

train_data[missing_data].isnull().sum().sort_values(ascending=False)

#train_data.isnull().sum().sort_values(ascending=False)
selected_features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

test_data[selected_features].isnull().sum()
# concatenating both data sets and dropping SalePrice from this

concat_data = pd.concat([train_data,test_data],ignore_index=True)

concat_data = concat_data.drop("SalePrice", 1)

ids = train_data["Id"]

concat_data.shape[0] # total no. of rows after concatenation
concat_data.shape
# check for null or empty data and remove features which have more than 1000 rows empty

missing_data = concat_data.columns[concat_data.isnull().any()]

concat_data[missing_data].isnull().sum().sort_values(ascending=False)
# dropping PoolQC, MiscFeature, Alley, Fence, FireplaceQu

concat_data=concat_data.drop("PoolQC", 1)

concat_data=concat_data.drop("MiscFeature", 1)

concat_data=concat_data.drop("Alley", 1)

concat_data=concat_data.drop("Fence", 1)

concat_data=concat_data.drop("FireplaceQu", 1)

concat_data=concat_data.drop("Id", 1)
# Numerical and categorical attriutes in data with missing values

Num_Col = concat_data.select_dtypes(include = ["float64", "int64"])

Numeric_Col_with_null_val = Num_Col.columns[Num_Col.isnull().any()].tolist()



Categorical_col = concat_data.select_dtypes(include = ["object"])

Categorical_col_with_null_val = Categorical_col.columns[Categorical_col.isnull().any()].tolist()



print("\nList of numerical (Float, Int) attributes with NaN values:\n",Numeric_Col_with_null_val)



print("\nList of categorical attributes with NaN values:\n",Categorical_col_with_null_val)
# Methods for quickly looking at data

# Looking at categorical values

def exploration(column):

    return concat_data[column].value_counts()



# assining the missing values

def assign_values(column, value):

    concat_data.loc[concat_data[column].isnull(),column] = value
# 1. LotFrontage correlation with LotArea

concat_data['LotFrontage'].corr(concat_data['LotArea'])
# create temp column SqrtLotArea and correlate with LotFrontage

concat_data['SqrtLotArea']=np.sqrt(concat_data['LotArea'])

concat_data['LotFrontage'].corr(concat_data['SqrtLotArea'])
temp_var = concat_data['LotFrontage'].isnull()

# My Assumption is that most of the plots are rectangular in shape so using square root

concat_data.LotFrontage[temp_var]=concat_data.SqrtLotArea[temp_var]

# deleting Temp column SqrtLotArea

del concat_data['SqrtLotArea']
# 2. MasVnrType

exploration('MasVnrType')
# "None" is the most frequent value, assign "None" for the Type, and 0.0 for the area.

assign_values('MasVnrType', 'None')

# 3. MasVnrArea

exploration('MasVnrArea')
assign_values('MasVnrArea', 0.0)
# 3. basement

basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

concat_data[basement_cols][concat_data['BsmtQual'].isnull()==True]
# cases where the categorical variables are NaN, 

#the numerical ones are 0. Which means there's no basement, so the categorical ones should also be set to "None".

for cols in basement_cols:

    if 'FinSF'not in cols:

        assign_values(cols,'None')
# 4. Electrical

exploration('Electrical')

# assign most frequent value

assign_values('Electrical','SBrkr')
# 5. Garages

garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

concat_data[garage_cols][concat_data['GarageType'].isnull()==True]
#Garage Imputation

for cols in garage_cols:

    if concat_data[cols].dtype==np.object:

        assign_values(cols,'None')

    else:

        assign_values(cols, 0)
#filling NA's with the mean of the column:

concat_data = concat_data.fillna(concat_data.mean())
# Checking again for any missing values

missing_data = concat_data.columns[concat_data.isnull().any()]

concat_data[missing_data].isnull().sum().sort_values(ascending=False)
exploration('Exterior1st')
# assign most frequent value

assign_values('Exterior1st','VinylSd')
exploration('Exterior2nd')

assign_values('Exterior2nd','VinylSd')
exploration('Functional')

assign_values('Functional','Typ')
exploration('KitchenQual')

assign_values('KitchenQual','TA')

exploration('MSZoning')

assign_values('MSZoning','RL')

exploration('SaleType')

assign_values('SaleType','WD')

exploration('Utilities')

assign_values('Utilities','AllPub')
# Checking again for any missing values

missing_data = concat_data.columns[concat_data.isnull().any()]

concat_data[missing_data].isnull().sum().sort_values(ascending=False)
concat_data_backup = concat_data

concat_data_backup.shape
# Handling the skewness

# 1. Target variable transform

train_data["SalePrice"] = np.log1p(train_data["SalePrice"])



#log transform skewed numeric features:

digi_features = concat_data.dtypes[concat_data.dtypes != "object"].index

skew_features = train_data[digi_features].apply(lambda x: skew(x.dropna())) # computing skewness

skew_features = skew_features[skew_features > 0.75] # getting features having skweness > 0.75

skew_features = skew_features.index # getting column names

concat_data[skew_features] = np.log1p(concat_data[skew_features]) # Now taking log1p transform of all skewed features
# Create dummy variables for categorical features

concat_data = pd.get_dummies(concat_data)
concat_data.shape
# spliting the normalized data back to test and train data

normalized_train_data = concat_data[:1460]

normalized_test_data = concat_data[1460:]
normalized_test_data.shape
# Normalized Sale Price data from training set

sale_price_data = train_data.SalePrice
from sklearn import linear_model

from sklearn.linear_model import Ridge, RidgeCV, LassoCV, LassoLarsCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score
# function returning cross-validation rmse error for evaluation of our models

def rmse(model):

    #cv = KFold(n_splits=5,shuffle=True,random_state=45)    

    rmse= np.sqrt(-cross_val_score(model, normalized_train_data, sale_price_data, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
# 1. Linear Regression Model

lm = linear_model.LinearRegression()

linear_regress_model = lm.fit(normalized_train_data, sale_price_data)

'Estimated intercept coeff: {0} and Number of coefficeients: {1}'.format(lm.intercept_, len(lm.coef_))
# 1.1 evaluation by cross validation

linear_regress_model_rmse = rmse(linear_regress_model)

print(linear_regress_model_rmse.mean())
lm.score(normalized_train_data, sale_price_data)
# Ridge

alpha_ridge = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

rmse_ridge = [rmse(Ridge(alpha = alpha)).mean() 

            for alpha in alpha_ridge]

print(rmse_ridge)
# for varying values of alpha we calculate rmse

cv_ridge = pd.Series(rmse_ridge, index = alpha_ridge)

cv_ridge.plot(title = "Ridge Validation")

plt.xlabel("alpha")

plt.ylabel("RMSE")
results={}

ridge_model = Ridge(alpha=5).fit(normalized_train_data, sale_price_data)

results["ridge"] = rmse(ridge_model)

results["ridge"].mean()
cv_ridge.min()
# Lasso

# We take a set of alphas

lasso_model = LassoCV(alphas = [5, 1, 0.1, 0.001, 0.0005]).fit(normalized_train_data, sale_price_data)

results["lasso"] = rmse(lasso_model)

mean_lasso = results["lasso"].mean()

print(mean_lasso)
# Lasso does some feature selection - setting coefficients of insignificant features to zero.

coef = pd.Series(lasso_model.coef_, index = normalized_train_data.columns)

"Lasso selected {0} variables and eliminated the other {1} variables ".format(str(sum(coef != 0)), str(sum(coef == 0)))
# simply plotting the coef

# cv_lasso = pd.Series(lasso_model.coef_, index = normalized_train_data.columns)

#plt.figure(figsize=(9,8))

#cv_lasso.plot(title = "Lasso Validation")

#plt.xlabel("coef")

#plt.ylabel("RMSE")
# plotting important coefficients

# The most important positive feature is GrLivArea - the above ground area by area square feet.

# Then a few other location and quality features contributed positively. 

# Some of the negative features make less sense and would be worth looking into more - 

# it seems like they might come from unbalanced categorical variables.

imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])

matplotlib.rcParams['figure.figsize'] = (8.0, 9.0)

imp_coef.plot(kind = "bar")

plt.title("Coefficients in the Lasso Model")
# Random Forest Regressor

# Each of the classifying tree is a decision tree

# It tries to build each tree using a different set of subset data and features

# final prediction is function of each prediction

from sklearn import ensemble

forest_et=ensemble.RandomForestRegressor(n_estimators=100, random_state=42)

forest_model = forest_et.fit(normalized_train_data, sale_price_data)

"Coefficient of determination on training set: {0}".format(forest_et.score(normalized_train_data, sale_price_data))
forest_rmse = rmse(forest_et)

results["forest"] = forest_rmse

forest_rmse_mean = forest_rmse.mean()

# print(forest_rmse_mean)

"Random Forest RMSE mean is: {0}".format(forest_rmse_mean)
# XGBoost

# comes from boosting, works on weights

# XGBoost is also known as ‘regularized boosting‘ technique.

# Parallel Processing, handles missing values, prediction is a combination of weighted input features

# we calculate the rmse mean for test and train set and plot it

import xgboost as xgb



dMat_train = xgb.DMatrix(normalized_train_data, label = sale_price_data)

dMat_test = xgb.DMatrix(normalized_test_data)

params = {"max_depth":20, "eta":0.1}

model = xgb.cv(params, dMat_train,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
# Fitting the model and predicting using xgboost

# colsample_bytree - fraction of columns to be randomly samples for each tree.

# gamma - specifies min value to split a node

# learning_rate - The learning parameter controls the magnitude of this change in the estimates.

# min_child_weight - minimum sum of weights of all observations required in a child.

# alpha - L1 regularization term on weight (analogous to Lasso regression)

# lambda - L2 regularization term on weights (analogous to Ridge regression)

# subsample - Denotes the fraction of observations to be randomly samples for each tree.

regr = xgb.XGBRegressor(colsample_bytree=0.4,

        gamma=0.045,

        learning_rate=0.07,

        max_depth=20,

        min_child_weight=1.5,

        n_estimators=300,

        reg_alpha=0.65,

        reg_lambda=0.45,subsample=0.95)

xgbmodel = regr.fit(normalized_train_data, sale_price_data)

# XGBoost rmse mean

xgboost_mean = rmse(xgbmodel).mean()

"XGBOOST rmse mean is: {0}".format(xgboost_mean)
results["xgboost"] = rmse(xgbmodel)
results["linear_regress"] = linear_regress_model_rmse
# XGboost score

"XGBOOST score is: {0}".format(regr.score(normalized_train_data, sale_price_data))
# predictions

lasso_predictions = np.expm1(lasso_model.predict(normalized_test_data))

ridge_predictions = np.expm1(ridge_model.predict(normalized_test_data))

forest_predictions = np.expm1(forest_model.predict(normalized_test_data))

xgb_predictions = np.expm1(xgbmodel.predict(normalized_test_data))
ridge_predictions
predictions = 0.5*lasso_predictions + 0.25*ridge_predictions + 0.2*xgb_predictions + 0.05*forest_predictions

# Generating csv file with predicted values

submission_file = pd.DataFrame({"id":test_data.Id, "SalePrice":predictions})

submission_file.to_csv("Linear_model.csv", index = False)
lm_predictions = np.expm1(linear_regress_model.predict(normalized_test_data))
# scatter plot showing xgb vs lasso

predictions = pd.DataFrame({"XGB":xgb_predictions, "Lasso":lasso_predictions})

predictions.plot(x = "XGB", y = "Lasso", kind = "scatter")



plt.plot([40000,500000], [40000, 500000], 'k-')
# scatter plot showing Forest vs lasso

predictions = pd.DataFrame({"Forest":forest_predictions, "Lasso":lasso_predictions})

predictions.plot(y = "Lasso", x = "Forest", kind = "scatter")

plt.plot([40000,500000], [40000, 500000], 'k-')
predictions = pd.DataFrame({"Ridge":ridge_predictions, "XGB":xgb_predictions})

predictions.plot(x = "Ridge", y = "XGB", kind = "scatter")

plt.plot([40000,500000], [40000, 500000], 'k-')
predictions = pd.DataFrame({"Forest":forest_predictions, "XGB":xgb_predictions})

predictions.plot(x = "Forest", y = "XGB", kind = "scatter")

plt.plot([40000,500000], [40000, 500000], 'k-')
#results["xgboost"]

#results["linear_regress"]

results
Ridge_vsLasso_rmse = pd.DataFrame({"Ridge":results['ridge'], "Lasso":results['lasso']})

Ridge_vsLasso_rmse.plot(x = "Ridge", y = "Lasso", kind = "scatter")
Xgb_vsLasso_rmse = pd.DataFrame({"XGB":results['xgboost'], "Lasso":results['lasso']})

Xgb_vsLasso_rmse.plot(x = "XGB", y = "Lasso", kind = "scatter")
forest_vs_Lasso_rmse = pd.DataFrame({"Forest":results['forest'], "Lasso":results['lasso']})

forest_vs_Lasso_rmse.plot(x = "Forest", y = "Lasso", kind = "scatter")
Ridge_vs_Xgb_rmse = pd.DataFrame({"Ridge":results['ridge'], "XGB":results['xgboost']})

Ridge_vs_Xgb_rmse.plot(x = "Ridge", y = "XGB", kind = "scatter")
forest_vs_Xgb_rmse = pd.DataFrame({"Forest":results['forest'], "XGB":results['xgboost']})

forest_vs_Xgb_rmse.plot(x = "Forest", y = "XGB", kind = "scatter")