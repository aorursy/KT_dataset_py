import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
H = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

H_T = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

H.head()
H.shape
H.info()
H[H.duplicated()]

#No duplicates
H[H.loc[:,~H.columns.isin(['SalePrice'])].duplicated()]

#No duplicates
Unique_data = H.apply(pd.Series.nunique)

Unique_data[Unique_data == 1]

#No single unique values in the dataset
Sum = H.isnull().sum().sort_values(ascending = False)

Percent = ((H.isnull().sum()*100)/H.count()[0]).sort_values(ascending = False)

NullValues = pd.concat([Sum, Percent], axis = 1, keys = ["Sum", "Percent"])

NullValues[NullValues.Sum > 0]
#Deleting columns that have more than 20% missing values and Id column

H.drop(['Id','Alley', 'PoolQC', 'Fence','MiscFeature','MiscVal','FireplaceQu','LotFrontage'], axis = 1, inplace = True)
Sum = H.isnull().sum().sort_values(ascending = False)

Percent = ((H.isnull().sum()*100)/H.count()[0]).sort_values(ascending = False)

NullValues = pd.concat([Sum, Percent], axis = 1, keys = ["Sum", "Percent"])

NullValues[NullValues.Sum > 0]
H[H["GarageArea"] == 0][['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual','GarageCond']]



# Missing values exist as there is no garage for these homes
H.fillna({'GarageType': 'NoGarage', 'GarageYrBlt': 0, 'GarageFinish': 'NoGarage', 'GarageQual': 'NoGarage','GarageCond': 'NoGarage'} , inplace = True)



#Filling appropriate values for nulls 
H[H["GarageArea"] == 0][['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual','GarageCond']]
H[H['TotalBsmtSF'] == 0][['BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum()



# Missing values exist as there is no basement for these homes
H[H['TotalBsmtSF'] > 0][['BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum()



# BsmtExposure and BsmtFinType2 have missing values though these homes have a basement
mode = H['BsmtExposure'].mode()[0]

H.loc[(H['TotalBsmtSF'] > 0) & (H['BsmtExposure'].isnull()), 'BsmtExposure'] = mode



mode = H['BsmtFinType2'].mode()[0]

H.loc[(H['TotalBsmtSF'] > 0) & (H['BsmtFinType2'].isnull()), 'BsmtFinType2'] = mode



#Filling these nulls with mode
H.fillna({'BsmtQual': 'NoBasement', 'BsmtCond': 'NoBasement','BsmtExposure': 'NoBasement', 'BsmtFinType1': 'NoBasement', 'BsmtFinType2': 'NoBasement'} , inplace = True)





#Filling appropriate values for other nulls 
mode = H['Electrical'].mode()[0]

H['Electrical'].fillna(mode, inplace = True)

mode = H['MasVnrType'].mode()[0]

H['MasVnrType'].fillna(mode, inplace = True)

median = H['MasVnrArea'].median()

H['MasVnrArea'].fillna(median, inplace = True)



# Filling missing values in MasVnrArea,MasVnrType,Electrical with mode
Sum = H.isnull().sum().sort_values(ascending = False)

Percent = ((H.isnull().sum()*100)/H.count()[0]).sort_values(ascending = False)

NullValues = pd.concat([Sum, Percent], axis = 1, keys = ["Sum", "Percent"])

NullValues[NullValues.Sum > 0]



#No null values
H["AgeOfHouse"] = 2011 - H["YearBuilt"]

H["AgeOfRemod"] = 2011 - H["YearRemodAdd"]

H['AgeOfSell'] = 2011 - H['YrSold']

H['AgeOfGarage'] = 2011 - H['GarageYrBlt']

H.loc[H['AgeOfGarage'] > 100 , 'AgeOfGarage'] = 0

H.drop(['YearBuilt','YearRemodAdd','YrSold','GarageYrBlt'], axis = 1, inplace = True)



#Using age instead of year for better intution and ease
H['BsmtBath'] = H['BsmtFullBath'] + (0.5 * H['BsmtHalfBath'])

H['Bath'] = H['FullBath'] + (0.5 * H['HalfBath'])
H['TotalPorchArea'] = H['OpenPorchSF'] + H['EnclosedPorch'] + H['3SsnPorch'] + H['ScreenPorch']
numerical_columns = ['SalePrice','LotArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','TotalPorchArea','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MasVnrArea','AgeOfGarage', 'AgeOfHouse', 'AgeOfRemod','AgeOfSell']

categorical_columns = ['BsmtBath','Bath','BedroomAbvGr','BldgType','BsmtHalfBath','BsmtFullBath','Condition1','Condition2','Electrical','Exterior1st','Exterior2nd','Fireplaces','Foundation','FullBath','Functional','GarageCars','GarageFinish','GarageType','HalfBath','Heating','HouseStyle','KitchenAbvGr','LandContour','LandSlope','LotConfig','LotShape','MSSubClass','MSZoning','MasVnrType','MoSold','Neighborhood','PavedDrive','RoofMatl','RoofStyle','SaleCondition','SaleType','Street','TotRmsAbvGrd','Utilities']

ordinal_columns = [ "OverallQual","OverallCond","ExterQual","ExterCond","BsmtQual",'BsmtCond',"BsmtExposure","HeatingQC","KitchenQual","GarageQual","GarageCond", 'BsmtFinType1', 'BsmtFinType2','CentralAir']
H[ordinal_columns]
H['ExterQual'] = H['ExterQual'].map({'Po' : 1, 'Fa': 2, 'TA' : 3, 'Gd': 4 , 'Ex' : 5})

H['ExterCond'] = H['ExterCond'].map({'Po' : 1, 'Fa': 2, 'TA' : 3, 'Gd': 4 , 'Ex' : 5})

H['BsmtQual'] = H['BsmtQual'].map({'NoBasement' : 0, 'NA' : 0, 'Po' : 1, 'Fa': 2, 'TA' : 3, 'Gd': 4 , 'Ex' : 5})

H['BsmtCond'] = H['BsmtCond'].map({'NoBasement' : 0, 'NA' : 0, 'Po' : 1, 'Fa': 2, 'TA' : 3, 'Gd': 4 , 'Ex' : 5})

H['BsmtExposure'] = H['BsmtExposure'].map({'Gd' : 4, 'Av' : 3, 'Mn' : 2, 'No' : 1, 'NoBasement' : 0})

H['HeatingQC'] = H['HeatingQC'].map({'Po' : 1, 'Fa': 2, 'TA' : 3, 'Gd': 4 , 'Ex' : 5})

H['KitchenQual'] = H['KitchenQual'].map({'Po' : 1, 'Fa': 2, 'TA' : 3, 'Gd': 4 , 'Ex' : 5})

H['GarageQual'] = H['GarageQual'].map({'NoGarage' : 0, 'NA' : 0, 'Po' : 1, 'Fa': 2, 'TA' : 3, 'Gd': 4 , 'Ex' : 5})

H['GarageCond'] = H['GarageCond'].map({'NoGarage' : 0, 'NA' : 0, 'Po' : 1, 'Fa': 2, 'TA' : 3, 'Gd': 4 , 'Ex' : 5})

H['BsmtFinType1'] = H['BsmtFinType1'].map({'GLQ' : 6, 'ALQ' : 5, 'BLQ' : 4, 'Rec' : 3, 'LwQ' : 2, 'Unf' : 1, 'NoBasement' : 0})

H['BsmtFinType2'] = H['BsmtFinType2'].map({'GLQ' : 6, 'ALQ' : 5, 'BLQ' : 4, 'Rec' : 3, 'LwQ' : 2, 'Unf' : 1, 'NoBasement' : 0})

H['CentralAir'] = H['CentralAir'].map({'N' : 0, 'Y' : 1})
H[ordinal_columns].head(5)
H[ordinal_columns].isnull().sum()
H[categorical_columns].dtypes
for i in range(0, len(categorical_columns)):

    if (H[categorical_columns[i]].dtype == 'int64') | (H[categorical_columns[i]].dtype == 'float64'):

        H[categorical_columns[i]] = H[categorical_columns[i]].apply(str)

        

#Changing data type to string/object
H[categorical_columns].head()
H["BsmtBath"].value_counts()
H.loc[H["BsmtBath"] == '3.0', 'BsmtBath'] = '2.0'



#Merging 3 baths to 2 as there is only 1 record
H["BedroomAbvGr"].value_counts()
H.loc[H["BedroomAbvGr"] == '8', 'BedroomAbvGr'] = '6'

#Merging 8 to closer one 6
H["BsmtFullBath"].value_counts()
H.loc[H["BsmtFullBath"] == '3', 'BsmtFullBath'] = '2'

#Merging 3 to closer one 2
H["Condition1"].value_counts()

# Not merging as they seem to have an importance
H["Condition2"].value_counts()
H.loc[H["Condition2"].isin(['PosA','RRAn','RRAe']), 'Condition2'] = 'PosA_RRAn_RRAe'

#Merging 'PosA','RRAn','RRAe' to one field
H["Electrical"].value_counts()

# Not merging as they seem to have some importance
H["Exterior1st"].value_counts()
H.loc[H["Exterior1st"].isin(['Stone','BrkComm','CBlock','AsphShn','ImStucc']), 'Exterior1st'] = 'Other'

#Renaming 'Stone','BrkComm','CBlock','AsphShn','ImStucc' to other
H["Exterior2nd"].value_counts()
H.loc[H["Exterior2nd"].isin(['Stone','Brk Cmn','CBlock','AsphShn','ImStucc','Other']), 'Exterior2nd'] = 'Other'

# Renaming 'Stone','BrkComm','CBlock','AsphShn','ImStucc' to Other
H["Utilities"].value_counts()
H.drop(['Utilities'], axis = 1, inplace = True)

categorical_columns.remove('Utilities')



#Droping Utilities as it has 99% data as 1 unique value
H["Heating"].value_counts()
H.loc[H["Heating"] == 'Floor', 'Heating'] = 'OthW'

#Merging 'Floor' to 'OthW'
H["RoofMatl"].value_counts()
H.loc[H["RoofMatl"].isin(['Roll','Membran','Metal','ClyTile']), 'RoofMatl'] = 'Other'

#Clubbing 'Roll','Membran','Metal','ClyTile' to other
H["TotRmsAbvGrd"].value_counts()
H.loc[H["TotRmsAbvGrd"] == '2', 'TotRmsAbvGrd'] = '3'

H.loc[H["TotRmsAbvGrd"] == '14', 'TotRmsAbvGrd'] = '12'

#Merging outliers to closer values
numerical_columns_1 = ['SalePrice','LotArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF','TotalPorchArea','MasVnrArea','AgeOfGarage', 'AgeOfHouse', 'AgeOfRemod','AgeOfSell']

plt.figure(figsize=(15,60))

for i in range(0, len(numerical_columns_1)):

    plt.subplot(12,2,(i+1))

    sns.distplot(H[numerical_columns_1[i]])
from scipy import stats



# correcting target variable

H['SalePrice'], fitted_lambda = stats.boxcox(H['SalePrice'])



#Correcting some normally distributed but skewed numerical data

H['LotArea'], fitted_lambda = stats.boxcox(H['LotArea'])

H['1stFlrSF'], fitted_lambda = stats.boxcox(H['1stFlrSF'])

H['GrLivArea'], fitted_lambda = stats.boxcox(H['GrLivArea'])



H['TotalBsmtSF'].describe()
sns.distplot(H['1stFlrSF'])
sns.distplot(H['GrLivArea'])
plt.figure(figsize=(15,15))

correlation = H[numerical_columns].corr()

sns.heatmap(correlation, annot = True)
plt.figure(figsize=(15,60))

for i in range(0, len(categorical_columns)):

    plt.subplot(20,2,(i+1))

    sns.boxplot(data = H, x = categorical_columns[i], y = 'SalePrice'  )



#Plotting all categorical with box plot to ponder and check for any obvious issues with data    
plt.figure(figsize=(15,60))

for i in range(0, len(numerical_columns)):

    plt.subplot(12,2,(i+1))

    sns.scatterplot(data = H, x = numerical_columns[i], y = 'SalePrice'  )

    

#Ploting all numerical data in scatter plots to ponder
plt.figure(figsize=(15,35))

for i in range(0, len(ordinal_columns)):

    plt.subplot(7,2,(i+1))

    sns.barplot(data = H, x = ordinal_columns[i], y = 'SalePrice'  )

    

#Ploting all ordinal values with Sala price in a bar graph 

# Sale price doesnt seem to change much with any ordinal data
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')

from sklearn import linear_model, metrics

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFE

y = H['SalePrice']



X = H.drop(['SalePrice'], axis = 1)
X.shape
House_Dummies = pd.get_dummies(H[categorical_columns], drop_first = True)

House_Dummies.head()
len(categorical_columns)
X = X.drop(categorical_columns, axis = 1)

X = pd.concat([X, House_Dummies], axis=1)
X.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 100)
scaler = StandardScaler()

numerical_columns.remove('SalePrice')

X_train[numerical_columns+ordinal_columns] = scaler.fit_transform(X_train[numerical_columns+ordinal_columns])

X_test[numerical_columns+ordinal_columns] = scaler.transform(X_test[numerical_columns+ordinal_columns])
X_train.shape
X_test.shape
params = {'alpha': [0.00001,0.00005,0.0001, 0.0005,0.001,0.01, 0.02]}

#arams = {'alpha': [0.1, 1,10,100,200,300,500,1000]}





lasso = Lasso()





folds = 5

#Taking 5 folds for Cross validation



model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
alpha =0.001

lasso = Lasso(alpha=alpha)  

lasso.fit(X_train, y_train) 
Lasso_coef = pd.DataFrame({"Feature":X_train.columns.tolist(),"Coefficients":lasso.coef_})

Lasso_coef[Lasso_coef['Coefficients'] != 0 ].sort_values(by = "Coefficients" , ascending = False).count()



#Feature selection is done by Lasso and features narrowed down to 46
y_test_lasso_predict = lasso.predict(X_test)

y_train_lasso_predict = lasso.predict(X_train)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_lasso_predict))

print(metrics.r2_score(y_true=y_train, y_pred=y_train_lasso_predict))


params = {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 10, 100, 1000]}





ridge = Ridge()



# cross validation with 5 folds

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
alpha =100

ridge = Ridge(alpha=alpha)  

ridge.fit(X_train, y_train) 
y_test_ridge_predict = ridge.predict(X_test)

y_train_ridge_predict = ridge.predict(X_train)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_ridge_predict))

print(metrics.r2_score(y_true=y_train, y_pred=y_train_ridge_predict))
Ridge_coef = pd.DataFrame({"Feature":X_train.columns.tolist(),"Coefficients":ridge.coef_})

Ridge_coef[Ridge_coef['Coefficients'] != 0 ].sort_values(by = "Coefficients" , ascending = False).count()

#No feature selection happened
params = {'alpha': [0,0.0001, 0.0005, 0.001, 0.01]}



elasticnet = ElasticNet()



# cross validation

model_cv = GridSearchCV(estimator = elasticnet, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
alpha =0.001

elasticnet = ElasticNet(alpha=alpha)  

elasticnet.fit(X_train, y_train) 
y_test_elasticnet_predict = elasticnet.predict(X_test)

y_train_elasticnet_predict = elasticnet.predict(X_train)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_elasticnet_predict))

print(metrics.r2_score(y_true=y_train, y_pred=y_train_elasticnet_predict))
elasticnet_coef = pd.DataFrame({"Feature":X_train.columns.tolist(),"Coefficients":elasticnet.coef_})

elasticnet_coef[elasticnet_coef['Coefficients'] != 0 ].sort_values(by = "Coefficients" , ascending = False).count()

#Feature selection happened but not as good as Lasso
Lasso_coef = Lasso_coef[Lasso_coef['Coefficients'] != 0 ].sort_values(by = "Coefficients" , ascending = False).reset_index()
Lasso_coef.drop(['index'], axis = 1, inplace = True)
Lasso_coef.head(10)
Lasso_coef.tail(10)
Lasso_coef['Feature'].to_list()


plt.figure(figsize=(15,15))

sns.barplot(x="Coefficients", y="Feature", data=Lasso_coef, palette="vlag")

plt.xlabel("Feature Importance")

plt.tight_layout()