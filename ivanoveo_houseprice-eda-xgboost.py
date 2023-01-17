import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_train.head(3)
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test.head(3)
df_train.shape
df_test.shape
y = df_train.SalePrice
y
df_all = pd.concat((df_test, df_train)).reset_index(drop=True)
df_all.drop('SalePrice', axis=1, inplace=True)
df_all.head(3)
categorical_features = [feature for feature in df_all.columns 
                        if df_all[feature].dtype == 'O']

print(f'Number of categorical variables: {len(categorical_features)}')

df_train[categorical_features].head(3)
categorical_with_nan = [feature for feature in df_all[categorical_features]
                       if df_all[feature].isnull().any()]
len(categorical_with_nan)
for feature in categorical_with_nan:
    print(f'{feature}: \t {np.around(df_all[feature].isnull().mean()*100, 2)}%  missing value')
numerical_features = [feature for feature in df_all.columns 
                      if df_all[feature].dtypes != 'O']

print('Number of numerical features: ', len(numerical_features))

df_all[numerical_features].head(3)
numerical_with_nan = [feature for feature in df_all[numerical_features] 
                      if df_all[feature].isnull().sum()
                      and df_all[feature].dtypes != 'O']

numerical_with_nan
for feature in numerical_with_nan:
    print(f'{feature}: \t {np.around(df_all[feature].isnull().mean()*100, 2)}% missing value')
discrete_features = [feature for feature in numerical_features 
                    if len(df_all[feature].unique())<25 
                    and feature not in ['Id']]

print('Discrete Feature Count: ', len(discrete_features))

df_all[discrete_features].head(3)
continuous_features = [feature for feature in numerical_features 
                      if feature not in discrete_features + ['Id']]

print(f'Continuous Features Count:  {len(continuous_features)}')

df_all[continuous_features].head(3)
for feature in continuous_features:
    df_all = df_all.copy()
    df_all[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title('Count vs ' + feature)
    plt.show()
def display_only_missing(df):
    all_data_na = (df.isnull().sum() / len(df)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    print(missing_data)
display_only_missing(df_train)
df_train.PoolQC.value_counts()
df_train.drop("PoolQC", axis=1, inplace=True)
df_test.drop("PoolQC", axis=1, inplace=True)
df_train.MiscFeature.value_counts()
df_train.drop("MiscFeature", axis=1, inplace=True)
df_test.drop("MiscFeature", axis=1, inplace=True)
df_train.Alley.value_counts()
df_train.drop("Alley", axis=1, inplace=True)
df_test.drop("Alley", axis=1, inplace=True)
df_train.Fence.value_counts()
df_train.drop("Fence", axis=1, inplace=True)
df_test.drop("Fence", axis=1, inplace=True)
df_train.FireplaceQu.value_counts()
import seaborn as sns

sns.countplot(df_train["FireplaceQu"])
sns.boxplot(data=df_train, x="SalePrice", y="FireplaceQu")
df_train["FireplaceQu"] = df_train["FireplaceQu"].fillna(0)
df_test["FireplaceQu"] = df_test["FireplaceQu"].fillna(0)
sns.boxplot(data=df_train, x="SalePrice", y="FireplaceQu")
sns.regplot(data=df_train, x="SalePrice",y="LotFrontage")
df_train["LotFrontage"] = df_train["LotFrontage"].fillna( df_train["LotFrontage"].median())
df_test["LotFrontage"] = df_test["LotFrontage"].fillna( df_train["LotFrontage"].median())
df_train.GarageQual.value_counts()
sns.boxplot(data=df_train, x="SalePrice", y="GarageQual")
df_train.drop("GarageQual", axis=1, inplace=True)
df_test.drop("GarageQual", axis=1, inplace=True)
df_train.GarageFinish.value_counts()
sns.boxplot(data=df_train, x="SalePrice", y="GarageFinish")
df_train["GarageFinish"] = df_train["GarageFinish"].fillna("NoGarage")
df_test["GarageFinish"] = df_test["GarageFinish"].fillna("NoGarage")
sns.boxplot(data=df_train, x="SalePrice", y="GarageFinish")
df_train.GarageCond.value_counts()
df_train["GarageCond"] = df_train["GarageCond"].fillna("NoGarage")
sns.boxplot(data=df_train, x="SalePrice", y="GarageCond")
df_train.drop("GarageCond", axis=1, inplace=True)
df_test.drop("GarageCond", axis=1, inplace=True)
sns.distplot(df_train.GarageYrBlt)
sns.regplot(data=df_train,x="SalePrice",y="GarageYrBlt")
#Заполним минимумом
df_train["GarageYrBlt"] = df_train["GarageYrBlt"].fillna(df_train.GarageYrBlt.min())
df_test["GarageYrBlt"] = df_test["GarageYrBlt"].fillna(df_train.GarageYrBlt.min())
df_train.GarageType.value_counts()
sns.boxplot(data=df_train, x="SalePrice", y="GarageType")
df_train["GarageType"] = df_train["GarageType"].fillna("NoGarage")
df_test["GarageType"] = df_test["GarageType"].fillna("NoGarage")
sns.boxplot(data=df_train, x="SalePrice", y="GarageType")
df_train.BsmtQual.value_counts()
sns.boxplot(data=df_train, x="SalePrice", y="BsmtQual")
df_train["BsmtQual"] = df_train["BsmtQual"].fillna("NoBsmt")
df_test["BsmtQual"] =df_test["BsmtQual"].fillna("NoBsmt")
df_train.drop("BsmtCond", axis=1, inplace=True)
df_test.drop("BsmtCond", axis=1, inplace=True)
df_train["BsmtExposure"] = df_train["BsmtExposure"].fillna("NoBsmt")
df_test["BsmtExposure"] =df_test["BsmtExposure"].fillna("NoBsmt")
df_train.BsmtFinType1.value_counts()
sns.boxplot(data=df_train, x="SalePrice", y="BsmtFinType1")
df_train["BsmtFinType1"] = df_train["BsmtFinType1"].fillna("NoBsmt")
df_test["BsmtFinType1"] =df_test["BsmtFinType1"].fillna("NoBsmt")
df_train.drop("BsmtFinType2", axis=1, inplace=True)
df_test.drop("BsmtFinType2", axis=1, inplace=True)
df_train.MasVnrType.value_counts()
sns.boxplot(data=df_train, x="SalePrice", y="MasVnrType")
df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")
df_test["MasVnrType"] =df_test["MasVnrType"].fillna("None")
df_train["Electrical"] = df_train["Electrical"].fillna("SBrkr")
df_test["Electrical"] =df_test["Electrical"].fillna("SBrkr")
df_train.MSZoning.value_counts()
sns.boxplot(data=df_train, x="SalePrice", y="MSZoning")
df_train["MSZoning"] = df_train["MSZoning"].fillna("RL")
df_test["MSZoning"] = df_test["MSZoning"].fillna("RL")
df_test.Functional.value_counts()
df_train.Functional.value_counts()
sns.boxplot(data=df_train, x="SalePrice", y="Functional")
df_test["Functional"] =df_test["Functional"].fillna("Typ")
df_train["Functional"].isnull().any()
df_test["BsmtFullBath"] =df_test["BsmtFullBath"].fillna(0)
df_test["BsmtHalfBath"] =df_test["BsmtHalfBath"].fillna(0)
df_test.Utilities.value_counts()
df_train.Utilities.value_counts()
df_test.SaleType.value_counts()
df_train.SaleType.value_counts()
sns.boxplot(data=df_train, x="SalePrice", y="SaleType")
df_test["SaleType"] =df_test["SaleType"].fillna("WD")
sns.distplot(df_test.GarageArea)
#Заполним минимумом.

df_test["GarageArea"] =df_test["GarageArea"].fillna(df_test.GarageArea.min())
df_test["GarageCars"] =df_test["GarageCars"].fillna(df_test.GarageCars.min())
df_test.KitchenQual.value_counts()
df_test["KitchenQual"] =df_test["KitchenQual"].fillna("TA")
sns.distplot(df_test.TotalBsmtSF)
df_test["TotalBsmtSF"] =df_test["TotalBsmtSF"].fillna(df_test.TotalBsmtSF.min())
df_test["BsmtUnfSF"] =df_test["BsmtUnfSF"].fillna(df_test.BsmtUnfSF.min())
df_test["BsmtFinSF2"] =df_test["BsmtFinSF2"].fillna(df_test.BsmtFinSF2.min())
df_test["BsmtFinSF1"] =df_test["BsmtFinSF1"].fillna(df_test.BsmtFinSF1.min())
df_test["Exterior1st"] =df_test["Exterior1st"].fillna("VinlSd")
df_test["Exterior2nd"] =df_test["Exterior2nd"].fillna("VinlSd")
df_train.columns[df_train.dtypes == "object"]
from sklearn.preprocessing import LabelEncoder
for col in df_train.columns[df_train.dtypes == "object"]:
    df_train[col] = df_train[col].factorize()[0]
df_train = df_train.drop(["Id"], axis=1)
for col in df_test.columns[df_test.dtypes == "object"]:
    df_test[col] = df_test[col].factorize()[0]
X = df_train.drop('SalePrice', axis=1)
X.head(3)
y = df_train.SalePrice
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33 ,random_state = 42)
from xgboost.sklearn import XGBRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
random_grid={'learning_rate':[0.001,0.01],
            'max_depth':[10,30],
            'n_estimators':[200,300],
            'subsample':[0.5,0.7]
}
xgb = XGBRegressor(objective='reg:linear')
grid_search=GridSearchCV(estimator=xgb,param_grid = random_grid,cv = 3, n_jobs = -1, verbose = 2,scoring='neg_mean_squared_error')
grid_search.fit(X_train,y_train)
print("\nGrid Search Best parameters set :")
print(grid_search.best_params_)
#Используем mse для оценки модели:
pred =grid_search.predict(X_test)
mse = np.mean((y_test - pred)**2)
print(f'MSE:{mse}')
sns.scatterplot(y_test,pred)
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.show()
