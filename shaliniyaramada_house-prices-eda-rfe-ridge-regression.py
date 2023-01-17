# Supress Warnings



import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

import datetime



from sklearn import linear_model

from sklearn.linear_model import LinearRegression



from sklearn.linear_model import Ridge,RidgeCV

from sklearn.linear_model import Lasso,LassoCV

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.metrics import explained_variance_score



import os

from sklearn.preprocessing import scale 



# hide warnings

import warnings

warnings.filterwarnings('ignore')

# Importing train.csv

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.shape
train.describe()
test.describe()

# Targt Variable

train.SalePrice.describe()
sns.distplot(train['SalePrice']);

print('Skewness:',train['SalePrice'].skew())

print('Kurtosis:',train['SalePrice'].kurt())
train['SalePrice_log'] = np.log1p(train['SalePrice'])

sns.distplot(train['SalePrice_log']);

print('Skewness:',train['SalePrice_log'].skew())

print('Kurtosis:',train['SalePrice_log'].kurt())

train.drop('SalePrice',axis = 1,inplace = True)
train['SalePrice_log'].head()
# #finding skewness and kurtosis for all

# for col in num_cols:

#     print('{:15}'.format(col), 

#           'Skewness: {:05.2f}'.format(train[col].skew()) , 

#           '   ' ,

#           'Kurtosis: {:06.2f}'.format(train[col].kurt())  

#          )
#Null Columns

train.columns[train.isnull().sum() > 0 ]
test.columns[test.isnull().sum() > 0 ]
print(test.MasVnrType.head(10))

print(train.MasVnrType.head(10))



#There are "None" in data, so replacing with null



train = train.replace('None',np.nan)

test =test.replace('None',np.nan)

#SUm of nulls

train.loc[:, train.isnull().sum() > 0].isnull().sum().sort_values()



test.loc[:, test.isnull().sum() > 0].isnull().sum().sort_values()

test.drop(["Id"],axis = 1,inplace = True)

train.drop(["Id"],axis = 1,inplace = True)

#dropping columns which have more null dATA+

train.drop(["Fence","Alley","MiscFeature","PoolQC","FireplaceQu","MasVnrType"],axis = 1,inplace =True)

test.drop(["Fence","Alley","MiscFeature","PoolQC","FireplaceQu","MasVnrType"],axis = 1,inplace =True)
#Deciding about LotFrontage

test['LotFrontage'].describe()
train.LotFrontage.fillna(train['LotFrontage'].median(),inplace = True)

test.LotFrontage.fillna(test['LotFrontage'].median(),inplace = True)
test.loc[:, test.isnull().sum() > 0].isnull().sum().sort_values()

#Replace all the empty garage with "no" as it doesn't have any garage

train.GarageType.fillna("no",inplace = True)

train.GarageYrBlt.fillna(0,inplace = True)

train.GarageFinish.fillna("no",inplace = True)

train.GarageQual.fillna("no",inplace = True)

train.GarageCond.fillna("no",inplace = True)



#Replace all the empty garage with "no" as it doesn't have any garage

test.GarageType.fillna("no",inplace = True)

test.GarageYrBlt.fillna(0,inplace = True)

test.GarageFinish.fillna("no",inplace = True)

test.GarageQual.fillna("no",inplace = True)

test.GarageCond.fillna("no",inplace = True)
test.loc[:, test.isnull().sum() > 0].isnull().sum().sort_values()

train.BsmtQual.fillna("no",inplace = True)

train.BsmtCond.fillna("no",inplace = True)

train.BsmtFinType1.fillna("no",inplace = True)

train.BsmtExposure.fillna("no",inplace = True)

train.BsmtFinType2.fillna("no",inplace = True)





test.BsmtQual.fillna("no",inplace = True)

test.BsmtCond.fillna("no",inplace = True)

test.BsmtFinType1.fillna("no",inplace = True)

test.BsmtExposure.fillna("no",inplace = True)

test.BsmtFinType2.fillna("no",inplace = True)

train.loc[:, train.isnull().sum() > 0].isnull().sum().sort_values()



train['Electrical'].mode().iloc[0] 

train.MasVnrArea.fillna(0,inplace = True)

train.Electrical.fillna(train['Electrical'].mode().iloc[0],inplace = True)



test.MasVnrArea.fillna(0,inplace = True)

test.Electrical.fillna(test['Electrical'].mode().iloc[0],inplace = True)

test.loc[:, test.isnull().sum() > 0].isnull().sum().sort_values()

test.ExterQual.value_counts()
test.MasVnrArea.fillna(0,inplace = True)

test.MSZoning.fillna(test.MSZoning.mode().iloc[0],inplace = True)

test.Functional.fillna(test.Functional.mode().iloc[0],inplace = True)

test.BsmtFullBath.fillna(test.BsmtFullBath.mode().iloc[0],inplace = True)

test.BsmtHalfBath.fillna(test.BsmtHalfBath.mode().iloc[0],inplace = True)



test.BsmtFinSF1.fillna(0,inplace = True)

test.BsmtFinSF2 .fillna(0,inplace = True)

test.BsmtUnfSF.fillna(0,inplace = True)

test.TotalBsmtSF.fillna(0,inplace = True)



test.BsmtUnfSF.fillna(0,inplace = True)

test.TotalBsmtSF.fillna(0,inplace = True)

test.GarageCars.fillna(0,inplace = True)

test.GarageArea.fillna(0,inplace = True)



test.SaleType.fillna(test.SaleType.mode().iloc[0],inplace = True)

test.KitchenQual.fillna(test.KitchenQual.mode().iloc[0],inplace = True)

test.Utilities.fillna(test.Utilities.mode().iloc[0],inplace = True)



test.Exterior1st.fillna(test.Exterior1st.mode().iloc[0],inplace = True)

test.Exterior2nd.fillna(test.Exterior2nd.mode().iloc[0],inplace = True)

print(train.shape)

print(test.shape)
train.head(20)
#seperating numerical data from categorical data for plotting correlation

num_cols = train.dtypes[train.dtypes != "object"].index

cat_cols = train.dtypes[train.dtypes == "object"].index
num_cols_test = test.dtypes[test.dtypes != "object"].index

cat_cols_test = test.dtypes[test.dtypes == "object"].index
#High correlations

cor_numVar = train[num_cols].corr()

corHigh = cor_numVar[abs(cor_numVar['SalePrice_log']) > 0.5]



plt.figure(figsize = (16, 10))

sns.heatmap(train[corHigh.index].corr(), annot = True, cmap="YlGnBu")

plt.show()
#Encode the ordinal data, starting 0 as low



# for cat in cat_cols_test:

#     print('--'*40)

#     print(cat)

#     print(test[cat].value_counts())
test.ExterQual.value_counts()
#Encode the ordinal data, starting 0 as low



#basement

train['BsmtFinType1'] = train['BsmtFinType1'].map({'no':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

train['BsmtFinType2'] = train['BsmtFinType2'].map({'no':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

train['BsmtExposure'] = train['BsmtExposure'].map({'no':0,'No':1,'Mn':2,'Av':3,'Gd':4})

train['BsmtCond'] = train['BsmtCond'].map({'no':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

train['BsmtQual'] = train['BsmtQual'].map({'no':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})



#Quality related

train['ExterQual'] = train['ExterQual'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

train['ExterCond'] = train['ExterCond'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

train['HeatingQC'] = train['HeatingQC'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

train['KitchenQual'] = train['KitchenQual'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

#train['FireplaceQu'] = train['FireplaceQu'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})



train['GarageQual'] = train['GarageQual'].map({'no':0,'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

train['GarageCond'] = train['GarageCond'].map({'no':0,'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})



train['GarageFinish'] = train['GarageFinish'].map({'no':0,'Unf':1,'RFn':2,'Fin':3})

train['GarageType'] = train['GarageType'].map({'no':0,'Detchd':1,'CarPort':2,'BuiltIn':3,

                                                         'Basment':4,'Attchd':5,'2Types':6})



#house realated

train['LotShape'] = train['LotShape'].map({'Reg':3,'IR1':2,'IR2':1,'IR3':0})

train['LandContour'] = train['LandContour'].map({'Low':0,'HLS':1,'Bnk':2,'Lvl':3})

train['Utilities'] = train['Utilities'].map({'ELO': 0,'NoSeWa':1,'NoSewr':2,'AllPub':3})

train['BldgType'] = train['BldgType'].map({'Twnhs':0,'TwnhsE':1,'Duplex':2,'2fmCon':3,'1Fam':4})

train['HouseStyle'] = train['HouseStyle'].map({'1Story':0,'1.5Fin':1,'1.5Unf':2,'2Story':3,

                                                         '2.5Fin':4,'2.5Unf':5,'SFoyer':6,'SLvl':7})

train['Functional'] = train['Functional'].map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,

                                                         'Mod':4,'Min2':5,'Min1':6,'Typ':7})



#others

train['LandSlope'] = train['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})

train['Street'] = train['Street'].map({'Grvl':0,'Pave':1})

#train['MasVnrType'] = train['MasVnrType'].map({'None':0,'BrkCmn':1,'BrkFace':2,'CBlock':3,'Stone':4})

train['CentralAir'] = train['CentralAir'].map({'N':0,'Y':1})

train['PavedDrive'] = train['PavedDrive'].map({'N':0,'P':1,'Y':2})



#Encode the ordinal data, starting 0 as low



#basement

test['BsmtFinType1'] = test['BsmtFinType1'].map({'no':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

test['BsmtFinType2'] = test['BsmtFinType2'].map({'no':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

test['BsmtExposure'] = test['BsmtExposure'].map({'no':0,'No':1,'Mn':2,'Av':3,'Gd':4})

test['BsmtCond'] = test['BsmtCond'].map({'no':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

test['BsmtQual'] = test['BsmtQual'].map({'no':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})



#Quality related

test['ExterQual'] = test['ExterQual'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

test['ExterCond'] = test['ExterCond'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

test['HeatingQC'] = test['HeatingQC'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

test['KitchenQual'] = test['KitchenQual'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

#train['FireplaceQu'] = train['FireplaceQu'].map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})



test['GarageQual'] = test['GarageQual'].map({'no':0,'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})

test['GarageCond'] = test['GarageCond'].map({'no':0,'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})



test['GarageFinish'] = test['GarageFinish'].map({'no':0,'Unf':1,'RFn':2,'Fin':3})

test['GarageType'] = test['GarageType'].map({'no':0,'Detchd':1,'CarPort':2,'BuiltIn':3,

                                                         'Basment':4,'Attchd':5,'2Types':6})



#house realated

test['LotShape'] = test['LotShape'].map({'Reg':3,'IR1':2,'IR2':1,'IR3':0})

test['LandContour'] = test['LandContour'].map({'Low':0,'HLS':1,'Bnk':2,'Lvl':3})

test['Utilities'] = test['Utilities'].map({'ELO': 0,'NoSeWa':1,'NoSewr':2,'AllPub':3})

test['BldgType'] = test['BldgType'].map({'Twnhs':0,'TwnhsE':1,'Duplex':2,'2fmCon':3,'1Fam':4})

test['HouseStyle'] = test['HouseStyle'].map({'1Story':0,'1.5Fin':1,'1.5Unf':2,'2Story':3,

                                                         '2.5Fin':4,'2.5Unf':5,'SFoyer':6,'SLvl':7})

test['Functional'] = test['Functional'].map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,

                                                         'Mod':4,'Min2':5,'Min1':6,'Typ':7})



#others

test['LandSlope'] = test['LandSlope'].map({'Gtl':0,'Mod':1,'Sev':2})

test['Street'] = test['Street'].map({'Grvl':0,'Pave':1})

#test['MasVnrType'] = test['MasVnrType'].map({'None':0,'BrkCmn':1,'BrkFace':2,'CBlock':3,'Stone':4})

test['CentralAir'] = test['CentralAir'].map({'N':0,'Y':1})

test['PavedDrive'] = test['PavedDrive'].map({'N':0,'P':1,'Y':2})



pd.options.display.float_format = '{:.2f}'.format

np.set_printoptions(suppress = False)

train.describe().transpose()
#Years converting to Age



CurrentYr = pd.to_numeric(datetime.datetime.now().year)



YearBuilt = pd.to_numeric(train.YearBuilt)

HouseAge = CurrentYr - YearBuilt

list(map(lambda x : x if x < 1000 else 0,HouseAge))

train["HouseAge"] = HouseAge

train.drop("YearBuilt",axis = 1,inplace = True)



YearRemoAdd = pd.to_numeric(train.YearRemodAdd)

YearRemoAdd = CurrentYr - YearRemoAdd

list(map(lambda x : x if x < 1000 else 0,YearRemoAdd))

train["RemodAge"] = YearRemoAdd

train.drop("YearRemodAdd",axis = 1,inplace = True)





GarYrblt = pd.to_numeric(train.GarageYrBlt)

GarageAge = CurrentYr - GarYrblt

list(map(lambda x : x if x < 1000 else 0,GarageAge))

train["GarageAge"] = GarageAge

train.drop(["GarageYrBlt"],axis =1,inplace = True)



YrSold = pd.to_numeric(train.YrSold)

SoldAge = CurrentYr - YrSold

list(map(lambda x : x if x < 1000 else 0,SoldAge))

train["SoldAge"] = SoldAge

train.drop(["SoldAge"],axis =1,inplace = True)

#Years converting to Age



CurrentYr = pd.to_numeric(datetime.datetime.now().year)



YearBuilt = pd.to_numeric(test.YearBuilt)

HouseAge = CurrentYr - YearBuilt

list(map(lambda x : x if x < 1000 else 0,HouseAge))

test["HouseAge"] = HouseAge

test.drop("YearBuilt",axis = 1,inplace = True)



YearRemoAdd = pd.to_numeric(test.YearRemodAdd)

YearRemoAdd = CurrentYr - YearRemoAdd

list(map(lambda x : x if x < 1000 else 0,YearRemoAdd))

test["RemodAge"] = YearRemoAdd

test.drop("YearRemodAdd",axis = 1,inplace = True)





GarYrblt = pd.to_numeric(test.GarageYrBlt)

GarageAge = CurrentYr - GarYrblt

list(map(lambda x : x if x < 1000 else 0,GarageAge))

test["GarageAge"] = GarageAge

test.drop(["GarageYrBlt"],axis =1,inplace = True)



YrSold = pd.to_numeric(test.YrSold)

SoldAge = CurrentYr - YrSold

list(map(lambda x : x if x < 1000 else 0,SoldAge))

test["SoldAge"] = SoldAge

test.drop(["SoldAge"],axis =1,inplace = True)

#Create some other features



train['Total_sqr_footage'] = (train['BsmtFinSF1'] + train['BsmtFinSF2'] +

                                 train['1stFlrSF'] + train['2ndFlrSF'])

train['Total_Bathrooms'] = (train['FullBath'] + (0.5*train['HalfBath']) + 

                               train['BsmtFullBath'] + (0.5*train['BsmtHalfBath']))



train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] +

                              train['EnclosedPorch'] + train['ScreenPorch'] +

                             train['WoodDeckSF'])



#simplified features

train['haspool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

train['has2ndfloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

train['hasgarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

train['hasbsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

train['hasfireplace'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#Create some other features



test['Total_sqr_footage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] +

                                 test['1stFlrSF'] + test['2ndFlrSF'])

test['Total_Bathrooms'] = (test['FullBath'] + (0.5*test['HalfBath']) + 

                               test['BsmtFullBath'] + (0.5*test['BsmtHalfBath']))



test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] +

                              test['EnclosedPorch'] + test['ScreenPorch'] +

                             test['WoodDeckSF'])



#simplified features

test['haspool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

test['has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

test['hasgarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

test['hasbsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

test['hasfireplace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
num_cols = train.dtypes[train.dtypes != "object"].index

num_cols

cat_cols = train.dtypes[train.dtypes == "object"].index

cat_cols

num_cols_test = test.dtypes[train.dtypes != "object"].index

num_cols_test

cat_cols_test = test.dtypes[train.dtypes == "object"].index

cat_cols_test



# Get the dummy variables for the feature

status = pd.get_dummies(train[cat_cols],drop_first = True)

status_test = pd.get_dummies(test[cat_cols],drop_first = True)

# Check what the dataset 'status' looks like

status_test.head()



train.drop(cat_cols, axis = 1,inplace = True)

test.drop(cat_cols, axis = 1,inplace = True)

# Add the results to the original housing dataframe

train = pd.concat([train, status], axis = 1)

test = pd.concat([test, status_test], axis = 1)

# Now let's see the head of our dataframe.

test.head()
train.head()
# Split the train dataset into X and y

y_train = train.pop('SalePrice_log')

X_train = train
num_cols = train.dtypes[train.dtypes != "object"].index

num_cols
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])



X_train.head()
test[num_cols_test] = scaler.fit_transform(test[num_cols_test])



test.head()
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 20)             # running RFE

rfe = rfe.fit(X_train, y_train)
col = X_train.columns[rfe.support_]

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
print(lm.summary())
X_train_new = X_train_rfe.drop(['MSZoning_FV'], axis=1)

X_train_new = X_train_new.drop(['BsmtFinSF1'], axis=1)

X_train_new = X_train_new.drop(['Total_sqr_footage'], axis=1)

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()

#Let's see the summary of our linear model

print(lm.summary())
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_new = X_train_new.drop(['RoofMatl_CompShg'], axis=1)

X_train_new = X_train_new.drop(['PoolArea'], axis=1)

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()

#Let's see the summary of our linear model

print(lm.summary())
#droping related to Roofmaterial

X_train_new = X_train_new.drop(['RoofMatl_WdShake'], axis=1)

X_train_new = X_train_new.drop(['RoofMatl_Tar&Grv'], axis=1)

X_train_new = X_train_new.drop(['RoofMatl_WdShngl'], axis=1)

X_train_new = X_train_new.drop(['RoofMatl_Membran'], axis=1)

X_train_new = X_train_new.drop(['RoofMatl_Metal'], axis=1)

X_train_new = X_train_new.drop(['RoofMatl_Roll'], axis=1)



#model5

# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_lm).fit()

#Let's see the summary of our linear model

print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#test set

X_train_new = X_train_new.drop(["const"],axis = 1)

X_test_new = test[X_train_new.columns]

print(X_train_new.shape)

print(X_test_new.shape)
X_train_new.columns
# Predict 

X_test_new = sm.add_constant(X_test_new)

y_pred = lm.predict(X_test_new)
alphas = 10**np.linspace(10,-2,100)*0.5

alphas



ridge = Ridge(normalize = True)

coefs = []



for a in alphas:

    ridge.set_params(alpha = a)

    ridge.fit(X_train_new, y_train)

    coefs.append(ridge.coef_)

    

np.shape(coefs)

print(coefs)
#Finding ALpha



ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)

ridgecv.fit(X_train_new, y_train)

ridgecv.alpha_

#fitting on model

ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)

ridge4.fit(X_train_new, y_train)



# Adding a constant variable 

X_test_new = test[X_train_new.columns]



y_pred_ridge = ridge4.predict(X_test_new)

pd.Series(ridge4.coef_, index = X_train_new.columns).sort_values(ascending = False)
results = pd.Series(y_pred_ridge)
converted_results = [(np.exp(x)) for x in [i for i in results]]
submit = pd.Series(converted_results)



submit
submit.to_csv('submit.csv', sep=',',index = False)
