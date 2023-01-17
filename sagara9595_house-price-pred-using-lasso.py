import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
housing_df_train = pd.read_csv(r"../input/house-prices-advanced-regression-techniques/train.csv")

housing_df_train.head()
housing_df_test = pd.read_csv(r"../input/house-prices-advanced-regression-techniques/test.csv")

housing_df_test.head()
housing_df_train.shape
housing_df_test.shape
housing_df = pd.concat((housing_df_train.loc[:,'MSSubClass':'SaleCondition'],

                      housing_df_test.loc[:,'MSSubClass':'SaleCondition']))
percentage = ((housing_df.isnull().sum() / housing_df.shape[0]) *100).sort_values(ascending=False)

Total = housing_df.isnull().sum().sort_values(ascending=False)

miss_data = pd.concat([Total, percentage], axis=1, keys=["Total", "Percentage"])

miss_data = miss_data[miss_data.Total >= 1]

miss_data
col_to_drop = miss_data.index

for col in col_to_drop:

    print(housing_df[col].value_counts(dropna = False))

    print("------------------------------------------------")
year_build = pd.concat([housing_df.YearBuilt, housing_df.GarageYrBlt], axis=1, keys=["House_Build", "Gar_Build"])

year_build.describe()
(year_build.House_Build == year_build.Gar_Build).sum() / year_build.shape[0] * 100
year_build["Gar_Build"] = np.where(year_build.Gar_Build.isnull(), year_build.House_Build, year_build.Gar_Build)
(year_build.House_Build == year_build.Gar_Build).sum() / year_build.shape[0] * 100
housing_df.GarageType = np.where(housing_df.GarageType.isnull(), "No Garage", housing_df.GarageType)

housing_df.GarageFinish = np.where(housing_df.GarageFinish.isnull(), "No Garage", housing_df.GarageFinish)

housing_df.GarageCond = np.where(housing_df.GarageCond.isnull(), "No Garage", housing_df.GarageCond)

housing_df.GarageQual = np.where(housing_df.GarageQual.isnull(), "No Garage", housing_df.GarageQual)

housing_df.GarageYrBlt = np.where(housing_df.GarageYrBlt.isnull(), housing_df.YearBuilt, housing_df.GarageYrBlt)
housing_df.BsmtFinType2 = np.where(housing_df.BsmtFinType2.isnull(), "No Basement", housing_df.BsmtFinType2)

housing_df.BsmtExposure = np.where(housing_df.BsmtExposure.isnull(), "No Basement", housing_df.BsmtExposure)

housing_df.BsmtQual = np.where(housing_df.BsmtQual.isnull(), "No Basement", housing_df.BsmtQual)

housing_df.BsmtFinType1 = np.where(housing_df.BsmtFinType1.isnull(), "No Basement", housing_df.BsmtFinType1)

housing_df.BsmtCond = np.where(housing_df.BsmtCond.isnull(), "No Basement", housing_df.BsmtCond)
percentage = ((housing_df.isnull().sum() / housing_df.shape[0]) *100).sort_values(ascending=False)

Total = housing_df.isnull().sum().sort_values(ascending=False)

miss_data = pd.concat([Total, percentage], axis=1, keys=["Total", "Percentage"])

miss_data = miss_data[miss_data.Total >= 1]

miss_data
lot_frontage = housing_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

lot_frontage.head()
housing_df["LotFrontage"] = housing_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
housing_df["PoolQC"] = housing_df["PoolQC"].fillna("No Pool")

housing_df["MiscFeature"] = housing_df["MiscFeature"].fillna("no misc feature")

housing_df["Alley"] = housing_df["Alley"].fillna("No Alley Access")

housing_df["Fence"] = housing_df["Fence"].fillna("No Fence")
housing_df.GarageType = np.where(housing_df.GarageType.isnull(), "No Garage", housing_df.GarageType)

housing_df.GarageFinish = np.where(housing_df.GarageFinish.isnull(), "No Garage", housing_df.GarageFinish)

housing_df.GarageCond = np.where(housing_df.GarageCond.isnull(), "No Garage", housing_df.GarageCond)

housing_df.GarageQual = np.where(housing_df.GarageQual.isnull(), "No Garage", housing_df.GarageQual)

housing_df.GarageYrBlt = np.where(housing_df.GarageYrBlt.isnull(), housing_df.YearBuilt, housing_df.GarageYrBlt)
housing_df.BsmtFinType2 = np.where(housing_df.BsmtFinType2.isnull(), "No Basement", housing_df.BsmtFinType2)

housing_df.BsmtExposure = np.where(housing_df.BsmtExposure.isnull(), "No Basement", housing_df.BsmtExposure)

housing_df.BsmtQual = np.where(housing_df.BsmtQual.isnull(), "No Basement", housing_df.BsmtQual)

housing_df.BsmtFinType1 = np.where(housing_df.BsmtFinType1.isnull(), "No Basement", housing_df.BsmtFinType1)

housing_df.BsmtCond = np.where(housing_df.BsmtCond.isnull(), "No Basement", housing_df.BsmtCond)
housing_df.FireplaceQu = np.where(housing_df.FireplaceQu.isnull(), "No Fireplace", housing_df.FireplaceQu)
percentage = ((housing_df.isnull().sum() / housing_df.shape[0]) *100).sort_values(ascending=False)

Total = housing_df.isnull().sum().sort_values(ascending=False)

miss_data = pd.concat([Total, percentage], axis=1, keys=["Total", "Percentage"])

miss_data = miss_data[miss_data.Total >= 1]

miss_data
housing_df["MasVnrType"] = housing_df["MasVnrType"].fillna("None")

housing_df["MasVnrArea"] = housing_df["MasVnrArea"].fillna(0)
housing_df['Electrical'] = housing_df['Electrical'].fillna(housing_df['Electrical'].mode()[0])
housing_df['MSZoning'] = housing_df['MSZoning'].fillna(housing_df['MSZoning'].mode()[0])

housing_df = housing_df.drop(['Utilities'], axis=1)

housing_df["Functional"] = housing_df["Functional"].fillna("Typ")

housing_df['Electrical'] = housing_df['Electrical'].fillna(housing_df['Electrical'].mode()[0])

housing_df['KitchenQual'] = housing_df['KitchenQual'].fillna(housing_df['KitchenQual'].mode()[0])

housing_df['Exterior1st'] = housing_df['Exterior1st'].fillna(housing_df['Exterior1st'].mode()[0])

housing_df['Exterior2nd'] = housing_df['Exterior2nd'].fillna(housing_df['Exterior2nd'].mode()[0])

housing_df['SaleType'] = housing_df['SaleType'].fillna(housing_df['SaleType'].mode()[0])

housing_df['MSSubClass'] = housing_df['MSSubClass'].fillna("None")
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    housing_df[col] = housing_df[col].fillna(0)
for col in ('GarageArea', 'GarageCars'):

    housing_df[col] = housing_df[col].fillna(0)
housing_df.isnull().sum().max()
#housing_df = housing_df.drop("Id",  axis=1)

categorical_var = housing_df.loc[:, housing_df.dtypes == np.object].columns

scal_var = housing_df.loc[:, housing_df.dtypes != np.object].columns
housing_df_dummy = pd.get_dummies(housing_df[categorical_var], drop_first=True)

housing_df_dummy.head()
housing_df_dummy.shape
housing_df = pd.concat([housing_df, housing_df_dummy], axis=1)

housing_df = housing_df.drop(categorical_var, axis=1)

housing_df.head()
X_train = housing_df[:housing_df_train.shape[0]]

X_test = housing_df[housing_df_test.shape[0]:-1]

y_train = housing_df_train.SalePrice
# model coefficients

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn import linear_model

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

# grid search cv

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



#Set up cross validation scheme

folds = KFold(n_splits=5, shuffle=True, random_state=4)



# specify the hyper parameters

param = {'alpha':[1.0, 5.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]}



model = Lasso()

model_cv = GridSearchCV(estimator=model, param_grid=param, 

                       scoring='r2', cv=folds, 

                        return_train_score=True, verbose=1)

model_cv.fit(X_train, y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plot

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('r2 score')

plt.vlines(x=15, ymax=1, ymin=0, colors='r', linestyles='--')

plt.show()
# model with optimal alpha

# lasso regression

lm = Lasso(alpha=15)

lm.fit(X_train, y_train)



# predict

y_train_pred = lm.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
# lasso model parameters

model_parameters = list(lm.coef_)

model_parameters.insert(0, lm.intercept_)

model_parameters = [round(x, 3) for x in model_parameters]

cols = X_train.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))
price_pred = lm.predict(X_test)
solution = pd.DataFrame({"id":housing_df_test.Id, "SalePrice":price_pred})

solution