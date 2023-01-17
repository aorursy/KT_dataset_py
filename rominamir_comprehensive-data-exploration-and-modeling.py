import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import missingno

from scipy import stats

# import researchpy as rp





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_raw  = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_raw = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



df_raw.head()

pd.options.display.max_rows = 81

df_raw.isnull().sum()





# sort a dataframe based on column names

df_raw = df_raw.sort_index(axis=1)

df_raw.describe()

df_raw.isnull().sum()
missingno.matrix(df_raw)
df = df_raw.copy()

# df_clean = df_clean.drop(['PoolQC', 'Fence'], axis = 1)



df["Alley"].fillna("No Alley", inplace = True)

df["PoolQC"].fillna("No Pool", inplace = True) 

df["MiscFeature"].fillna("Nothing", inplace = True) 

df["LotFrontage"].fillna("0", inplace = True) 

df["Fence"].fillna("No Fence", inplace = True) 

df["MasVnrType"].fillna("No MasVnr", inplace = True) 

df["MasVnrArea"].fillna("0", inplace = True) 

df["BsmtExposure"].fillna("No Basement", inplace = True) 

df["BsmtQual"].fillna("No Basement", inplace = True) 

df["BsmtCond"].fillna("No Basement", inplace = True) 

df["BsmtFinType1"].fillna("No Basement", inplace = True) 

df["BsmtFinType2"].fillna("No Basement", inplace = True) 

df["FireplaceQu"].fillna("No Fireplace", inplace = True) 

df["GarageType"].fillna("No Garage", inplace = True) 

df["GarageYrBlt"].fillna("0", inplace = True) 

df["GarageQual"].fillna("No Garage", inplace = True) 

df["GarageCond"].fillna("No Garage", inplace = True) 

df["GarageFinish"].fillna("No Garage", inplace = True) 



df["Electrical"].fillna("Mix", inplace = True) 



df.isnull().sum()

test_raw.isnull().sum()
test =test_raw.copy()



test["Alley"].fillna("No Alley", inplace = True)

test["PoolQC"].fillna("No Pool", inplace = True) 

test["MiscFeature"].fillna("Nothing", inplace = True) 

test["LotFrontage"].fillna("0", inplace = True) 

test["Fence"].fillna("No Fence", inplace = True) 

test["MasVnrType"].fillna("No MasVnr", inplace = True) 

test["MasVnrArea"].fillna("0", inplace = True) 

test["BsmtExposure"].fillna("No Basement", inplace = True) 

test["BsmtQual"].fillna("No Basement", inplace = True) 

test["BsmtCond"].fillna("No Basement", inplace = True) 

test["BsmtFinType1"].fillna("No Basement", inplace = True) 

test["BsmtFinType2"].fillna("No Basement", inplace = True) 

test["FireplaceQu"].fillna("No Fireplace", inplace = True) 

test["GarageType"].fillna("No Garage", inplace = True) 

test["GarageYrBlt"].fillna("0", inplace = True) 

test["GarageQual"].fillna("No Garage", inplace = True) 

test["GarageCond"].fillna("No Garage", inplace = True) 

test["GarageFinish"].fillna("No Garage", inplace = True) 



test.isnull().sum()
#All the vcategoricak columns

for col in ['MSZoning','Utilities','SaleType','KitchenQual',

            'Alley', 'LotFrontage', 'Street', 'LandContour', 

             'LotConfig',  'LandSlope','Neighborhood',

            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 

            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 

            'MasVnrType', 'MasVnrArea','ExterQual', 'ExterCond',

            'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

            'BsmtFinType1', 'BsmtFinType2', 'LotShape', 'GarageQual',

            'LotShape','GarageType','Functional','GarageFinish',

            'PoolQC', 'Fence','MiscFeature','SaleCondition',

            'Heating','HeatingQC','CentralAir','Electrical','FireplaceQu',

            'GarageYrBlt','GarageCond','PavedDrive']:

    test[col] = test[col].astype('category')

    #convertu=ing nume

    test[col] = test[col].cat.codes



    

test = test.sort_index(axis=1)
from sklearn.impute import KNNImputer



imputer = KNNImputer()

test_filled = imputer.fit_transform(test)

for col in ['Alley', 'MSZoning', 'LotFrontage', 'Street', 'LandContour', 

            'Utilities', 'LotConfig',  'LandSlope','Neighborhood',

            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 

            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 

            'MasVnrType', 'MasVnrArea','ExterQual', 'ExterCond',

            'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

            'BsmtFinType1', 'BsmtFinType2', 'LotShape', 'GarageQual',

            'LotShape','GarageType','KitchenQual','Functional','GarageFinish',

            'PoolQC', 'Fence','MiscFeature','SaleType','SaleCondition',

            'Heating','HeatingQC','CentralAir','Electrical','FireplaceQu',

            'GarageYrBlt','GarageCond','PavedDrive']:

    df[col] = df[col].astype('category')

    df[col] = df[col].cat.codes
plt.subplots(figsize=(25,20))



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))



sns.heatmap(df.corr(), annot = True, mask=mask)
g =sns.distplot(df['SalePrice'], label ='Sale Price')

g.set(xlim = (0,1000000));

xlabels = ['{:,.2f}'.format(x) + 'K' for x in g.get_xticks()/1000]

g.set_xticklabels(xlabels)

plt.title('Housing price distribution')
res = stats.probplot(df['SalePrice'], plot=plt)
plt.subplots(figsize=(10,7))

plt.title('subclassses Frequency')

sns.countplot(df['MSSubClass'])
fig, ax = plt.subplots(figsize=(15,7))

df.groupby(['MSSubClass']).count()['SalePrice'].plot(ax=ax)
pd.crosstab(df_raw.Fireplaces,df_raw.MSSubClass).plot(kind="bar",figsize=(15,7))

plt.title('subclassses and fireplace')



plt.xlabel('number of fireplaces')

plt.ylabel('Frequency')

# plt.savefig('heartDiseaseAndAges.png')

plt.show()

x1 = pd.Series(df['SalePrice'], name="$X_1$")

x2 = pd.Series(df['OverallQual'], name="$X_2$")



# Show the joint distribution using kernel density estimation

g = sns.jointplot(x1, x2, kind="kde", height=7, space=0, xlim = (0,800000) )

# visualize the relationship between the features and the response using scatterplots

sns.pairplot(df, x_vars=['GrLivArea','GarageArea',  'YearBuilt'],

             y_vars='SalePrice', size=7, aspect=0.7) # kind='reg'

sns.pairplot(df, x_vars=['BsmtFullBath','BsmtHalfBath' ,'FullBath', 'HalfBath'],

             y_vars='SalePrice', size=7, aspect=0.7) # kind='reg'
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, RandomForestClassifier

from sklearn.kernel_ridge import KernelRidge

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.model_selection import train_test_split

X = df.drop('SalePrice', axis= 1) #data

y = df.SalePrice #label



X_train, X_test, y_train, y_test = train_test_split(X, y , train_size = 0.70 , random_state =  90)



model_rf = RandomForestClassifier(random_state=30, n_estimators= 10);

model_rf.fit(X_train, y_train);
predict =model_rf.predict(X_test)



from sklearn import metrics

print ('Accuracy:', metrics.accuracy_score(y_test,predict))

print(model_rf.feature_importances_)
# Fitting Simple Linear Regression to the Training set

linreg = LinearRegression();

linreg.fit(X_train, y_train);
#print y_intercept

print('Y-intercept: ', linreg.intercept_)



coeff_df = pd.DataFrame(linreg.coef_, X.columns, columns=['Coefficient'])  

print('Model coef: ', coeff_df);







y_pred =linreg.predict(X_test)



#MAE

print('MAE: ',(metrics.mean_absolute_error(y_test,y_pred)))



#MSE

print('MSE: ', metrics.mean_squared_error(y_test,y_pred))



# RMSE

print('RSME: ', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# print the R-squared value for the model

print( 'Accuracy:', linreg.score(X, y))


df_training = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1 =df_training.head(25)

df1



df1.plot(kind='bar',figsize=(14,8))

plt.grid(which='major', linestyle='-', linewidth='0.5')

plt.grid(which='minor', linestyle=':', linewidth='0.5')

plt.show()

rfreg = RandomForestRegressor()

rfreg.fit(X_train, y_train)



print(rfreg.feature_importances_)



y_pred =rfreg.predict(X_test)

print( 'Accuracy:', rfreg.score(X, y))



#MAE

print('MAE: ',int((metrics.mean_absolute_error(y_test,y_pred))))



#MSE

print('MSE: ', int(metrics.mean_squared_error(y_test,y_pred)))



# RMSE

print('RSME: ',int( np.sqrt(metrics.mean_squared_error(y_test,y_pred))))



df_training2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df2=df_training2.head(25)

df2





df2.plot(kind='bar',figsize=(14,8))

plt.grid(which='major', linestyle='-', linewidth='0.5')

plt.grid(which='minor', linestyle=':', linewidth='0.5')

plt.show()







test_pred =rfreg.predict(test_filled)
submission = pd.DataFrame({

        "Id": test['Id'],

        "SalePrice": test_pred

    })

submission



submission.to_csv('..\\house_price\\submission.csv', index=False)