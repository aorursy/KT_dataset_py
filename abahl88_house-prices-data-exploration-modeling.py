#importing the standard python libraries for data exploration

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library

import matplotlib.pyplot as plt # data visualization library

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 

# plot displayed next to code
#Reading in our dataset

train_df = pd.read_csv('../input/train.csv') # reading in the training dataset.

test_df = pd.read_csv('../input/test.csv')# reading in the test dataset.

combined = pd.concat([train_df,test_df], axis=0) #combining train and test dataset for data preprocessing

combined.shape

train_df.shape # training dataset has 1460 observations
test_df.shape # test dataset has 1459 observations
combined.head() # read the first few rows of the dataset
#How many columns with different data types are there?

combined.get_dtype_counts()
combined['SalePrice'].describe() # checking how Sales Price Variable is distributed
#plot the distribution plot of SalePrices of the houses

plt.figure(figsize=(12,6))

sns.distplot(combined['SalePrice'].dropna() ,kde= False,bins=75 , rug = True ,color='purple')

sns.set(font_scale = 1.25)

plt.tight_layout()

plt.title('Distribution of Sale Price')
sp_corr = combined.corr()["SalePrice"]

sp_corr_sort = sp_corr.sort_values(axis = 0 , ascending = False)

sp_corr_sort[sp_corr_sort > 0.50]
corr = combined[["SalePrice","OverallQual","GrLivArea","GarageCars",

                  "GarageArea","TotalBsmtSF","1stFlrSF","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()
plt.figure(figsize=(18,10))

sns.heatmap(corr, linecolor= "white" , lw =1,annot=True)
sns.boxplot(y="SalePrice", x="OverallQual", data=combined)

plt.title('Overall Quality vs SalePrice')
sns.jointplot(y="SalePrice", x="GrLivArea", data=combined)

plt.title('Ground Living Area vs SalePrice')
sns.jointplot(y="SalePrice", x="GarageArea", data=combined)

plt.title('GarageArea vs SalePrice')
sns.jointplot(y="SalePrice", x="1stFlrSF", data=combined)

plt.title('1st Floor Surface Area vs SalePrice')
sns.boxplot(y="SalePrice", x="FullBath", data=combined)

plt.title('FullBath Area vs SalePrice')
sns.jointplot(y="SalePrice", x="YearBuilt", data=combined)

plt.title('YearBuilt vs SalePrice')
#Categorical variables

combined.select_dtypes(include=['object']).columns
#Numerical Columns

combined.select_dtypes(include=['float64', 'int64']).columns
from sklearn.preprocessing import Imputer



mean_imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)

median_imputer = Imputer(missing_values='NaN', strategy = 'median', axis=0)

mode_imputer = Imputer(missing_values='NaN', strategy = 'most_frequent', axis=0)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#Missing values in the columns

combined[combined.columns[combined.isnull().any()]].isnull().sum()
sns.countplot(x = 'Alley' , data = combined )
combined['Alley'].fillna('None',inplace = True)
combined[combined['BsmtCond'].isnull() == True][['BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'BsmtQual','BsmtFinSF1',

       'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF','TotalBsmtSF']]

#Categorical features 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'BsmtQual'



combined['BsmtQual'].fillna(value = 'None' , inplace = True)

combined['BsmtCond'].fillna(value = 'None' , inplace = True)

combined['BsmtExposure'].fillna(value = 'None' , inplace = True)

combined['BsmtFinType1'].fillna(value = 'None' , inplace = True)

combined['BsmtFinType2'].fillna(value = 'None' , inplace = True)
#Numerical Features 'BsmtCond','BsmtFinSF1','BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF','TotalBsmtSF'



combined['BsmtFinSF1'].fillna(value = 0 , inplace = True)

combined['BsmtFinSF2'].fillna(value = 0 , inplace = True)

combined['BsmtFullBath'].fillna(value = 0 , inplace = True)

combined['BsmtHalfBath'].fillna(value = 0 , inplace = True)

combined['BsmtUnfSF'].fillna(value = 0 , inplace = True)

combined['TotalBsmtSF'].fillna(value = 0 , inplace = True)
sns.countplot(x = 'Electrical' , data = combined)

combined['Electrical'].fillna(value = 'SBrkr' , inplace = True)
combined[combined['FireplaceQu'].isnull() == True][['Fireplaces','FireplaceQu']]

combined['FireplaceQu'].fillna(value = 'None' , inplace =  True)

combined[combined['GarageType'].isnull() == True][['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageArea','GarageCars']]
combined['GarageType'].fillna(value = 'None' , inplace = True)

combined['GarageYrBlt'].fillna(value = 'None' , inplace = True)

combined['GarageFinish'].fillna(value = 'None' , inplace = True)

combined['GarageQual'].fillna(value = 'None' , inplace = True)

combined['GarageCond'].fillna(value = 'None' , inplace = True)

combined['GarageArea'].fillna(value = 0 , inplace = True)

combined['GarageCars'].fillna(value = 0 , inplace = True)
sns.countplot(x = 'PoolQC' , data = combined)
combined[combined['PoolQC'].isnull() == True][['PoolQC','PoolArea']]
combined['PoolQC'].fillna(value = 'None' , inplace = True)
sns.distplot(combined['LotFrontage'].dropna() , bins =70)
combined['LotFrontage'] = combined['LotFrontage'].transform(lambda x: x.fillna(x.mode()[0]))
combined['MiscFeature'] = combined['MiscFeature'].fillna('None')

combined['Exterior1st'].fillna(value= 'None', inplace = True)

combined['Exterior2nd'].fillna(value= 'None', inplace = True)

combined['Functional'].fillna(value= 'None', inplace = True)

combined['KitchenQual'].fillna(value = 'None' , inplace = True)

combined['MSZoning'].fillna(value = 'None' , inplace = True)

combined['SaleType'].fillna(value = 'None' , inplace = True)

combined['Utilities'].fillna(value = 'None' , inplace = True)

combined["MasVnrType"] = combined["MasVnrType"].fillna('None')

combined["MasVnrArea"] = combined["MasVnrArea"].fillna(0)

combined["Fence"] = combined["Fence"].fillna('None')
combined[combined.columns[combined.isnull().any()]].isnull().sum()
#Skewed variables affect the performance of a Regression ML model so we do a Log transform to remove skewness

from scipy.stats import skew

#log transform the target:

combined["SalePrice"] = np.log1p(combined["SalePrice"])



#log transform skewed numeric features:

numeric_feats = combined.dtypes[combined.dtypes != "object"].index
skewed_feats = combined[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



combined[skewed_feats] = np.log1p(combined[skewed_feats])
combined.head()
categorical = combined.select_dtypes(exclude=['float64', 'int64'])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



labelEnc=LabelEncoder()



cat_vars=['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

       'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2',

       'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd',

       'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond',

       'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC',

       'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',

       'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood',

       'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition',

       'SaleType', 'Street', 'Utilities']



for col in cat_vars:

    combined[col]=labelEnc.fit_transform(combined[col])
combined.head()
# Year Columns

#'GarageYrBlt','YearBuilt','YearRemodAdd', 'YrSold'
combined['GarageYrBlt'].replace('None' , 100, inplace = True)
bins = [10,1960, 1980, 2000, 2017]

group_names = ['VeryOld', 'Old', 'Okay', 'New'] #Grouping into categories

combined['GarageYrBlt'] = pd.cut((combined['GarageYrBlt']), bins, labels=group_names)

combined['GarageYrBlt'].fillna('VeryOld', inplace = True)

combined['YearBuilt'] = pd.cut((combined['YearBuilt']), bins, labels=group_names)

combined['YearRemodAdd'] = pd.cut((combined['YearRemodAdd']), bins, labels=group_names)

combined['YrSold'] = pd.cut((combined['YrSold']), bins, labels=group_names)
from sklearn.preprocessing import LabelEncoder



labelEnc=LabelEncoder()



cat_vars=['GarageYrBlt','YearBuilt','YearRemodAdd', 'YrSold']



for col in cat_vars:

    combined[col]=labelEnc.fit_transform(combined[col])
combined.head()
New_Train = combined[:1460]

X_train = New_Train.drop('SalePrice',axis=1)

y_train = New_Train['SalePrice']
New_Train.shape
New_Test = combined[1460:]

X_test = New_Test.drop('SalePrice',axis=1)
X_test.shape
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score
#Defining a function to calculate the RMSE for each Cross validated fold

def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))

    return (rmse)
model_ridge = Ridge(alpha = 5).fit(X_train, y_train)
alphas = [0.0001,0.1,0.5,1,2,5,7,10]

rmse_cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

print(rmse_cv_ridge)
rmse_cv_ridge = pd.Series(rmse_cv_ridge, index = alphas)

rmse_cv_ridge.plot(title = "RMSE VS Alpha")
model_ridge = Ridge(alpha = 2).fit(X_train, y_train)
rmse_cv(model_ridge).mean()
ridge_preds = np.exp(model_ridge.predict(X_test))
from sklearn.linear_model import Lasso
model_lasso = Lasso().fit(X_train, y_train)
alphas = [0.00001,.0001,0.001,0.002,0.005,0.01]

rmse_cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]

print(rmse_cv_lasso)
rmse_cv_lasso = pd.Series(rmse_cv_lasso, index = alphas)

plt.figure(figsize=(10,4))

rmse_cv_lasso.plot(title = "RMSE VS Alpha")
model_lasso = Lasso(alpha = 0.001 , max_iter=1000).fit(X_train, y_train)
rmse_cv(model_lasso).mean()
#checking the magnitude of coefficients, which coefficients are most important.



predictors = X_train.columns



coef = pd.Series(model_lasso.coef_, index = X_train.columns)



imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])



plt.figure(figsize=(12,8))

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
lasso_preds = np.expm1(model_lasso.predict(X_test)) # reversing Log Transformation
Final = 0.5*lasso_preds + 0.5*ridge_preds # combining the models
submission = pd.DataFrame({

        "Id": X_test["Id"],

        "SalePrice": Final

    })
submission.to_csv("HousePrice.csv", index=False)