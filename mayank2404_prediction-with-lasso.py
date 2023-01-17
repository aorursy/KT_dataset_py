import numpy as np 
import pandas as pd 
import seaborn as sns
import os
import matplotlib.pyplot as plt
%matplotlib inline
train_d = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_d = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_d.info()
test_d.info()
ax = sns.heatmap(train_d.isnull(),yticklabels=False,cbar=False,fmt='.1f')
ax.set(xlabel='columns', ylabel='rows (white if null)')
plt.show()
# data imputation on training datframe:



train_d['Alley'].fillna('No', inplace=True)
train_d['BsmtQual'].fillna('No', inplace=True)
train_d['BsmtCond'].fillna('No', inplace=True)
train_d['BsmtExposure'].fillna('No Basement', inplace=True)
train_d['BsmtFinType1'].fillna('No', inplace=True)
train_d['BsmtFinType2'].fillna('No', inplace=True)
train_d['FireplaceQu'].fillna('No', inplace=True)
train_d['GarageType'].fillna('No', inplace=True)
train_d['GarageFinish'].fillna('No', inplace=True)
train_d['GarageQual'].fillna('No', inplace=True)
train_d['GarageCond'].fillna('No', inplace=True)
train_d['PoolQC'].fillna('No', inplace=True)
train_d['Fence'].fillna('No', inplace=True)
train_d['GarageCond'].fillna('No', inplace=True)




# data imputation on testing datframe:

test_d['Alley'].fillna('No', inplace=True)
test_d['BsmtQual'].fillna('No', inplace=True)
test_d['BsmtCond'].fillna('No', inplace=True)
test_d['BsmtExposure'].fillna('No Basement', inplace=True)
test_d['BsmtFinType1'].fillna('No', inplace=True)
test_d['BsmtFinType2'].fillna('No', inplace=True)
test_d['FireplaceQu'].fillna('No', inplace=True)
test_d['GarageType'].fillna('No', inplace=True)
test_d['GarageFinish'].fillna('No', inplace=True)
test_d['GarageQual'].fillna('No', inplace=True)
test_d['GarageCond'].fillna('No', inplace=True)
test_d['PoolQC'].fillna('No', inplace=True)
test_d['Fence'].fillna('No', inplace=True)
test_d['GarageCond'].fillna('No', inplace=True)
#check for outliers to proceed with imputation

sns.jointplot('LotFrontage','SalePrice',data = train_d) 
# remove rows with LotFrontage more than 300
#train_d = train_d[train_d.LotFrontage < 300].reset_index(drop = True)
#sns.jointplot('LotFrontage','SalePrice',data = train_d) 
train_d = train_d.drop(train_d[(train_d['LotFrontage'] > 300) & (train_d['SalePrice'] <400000)].index)
sns.jointplot('LotFrontage','SalePrice',data = train_d) 
sns.jointplot('GrLivArea','SalePrice',data = train_d) 
#there is just one outlier which affects the linear rshp between price and live area
#train_d = train_d[(train_d.GrLivArea< 4000) & (train_d.SalePrice>500000)].reset_index(drop = True)
train_d = train_d.drop(train_d[(train_d['GrLivArea'] > 4000) & (train_d['SalePrice'] <400000)].index)
train_d.count()
# imputation on LotFrontage and GarageYearbult

train_d['LotFrontage'] = train_d['LotFrontage'].fillna(train_d['LotFrontage'].median())
train_d['GarageYrBlt'] = train_d['GarageYrBlt'].fillna(train_d['GarageYrBlt'].mean())
test_d['LotFrontage'] = test_d['LotFrontage'].fillna(test_d['LotFrontage'].median())
test_d['GarageYrBlt'] = test_d['GarageYrBlt'].fillna(test_d['GarageYrBlt'].mean())
train_d.isnull().values.any()
# Drop MiscFeature as it has more than 90% missing values and not sure what it basically means, and Id as it is not needed
test_IdBackup=test_d
train_d=train_d.drop(['Id','MiscFeature','MasVnrArea'],axis=1)
test_d=test_d.drop(['Id','MiscFeature','MasVnrArea'],axis=1)
correlations_value_list=train_d.corr()['SalePrice'].sort_values()
print(correlations_value_list)
# data transformation for training dataset


#one hot vector encoding

col1 = pd.get_dummies(train_d.MSSubClass, prefix='MSSubClass')
del train_d['MSSubClass']
train_d= pd.concat([train_d, col1], axis=1)

col2 = pd.get_dummies(train_d.MSZoning, prefix='MSZoning')
del train_d['MSZoning']
train_d= pd.concat([train_d, col2], axis=1)


col3 = pd.get_dummies(train_d.Street, prefix='Street')
del train_d['Street']
train_d= pd.concat([train_d, col3], axis=1)


col4 = pd.get_dummies(train_d.Alley, prefix='Alley')
del train_d['Alley']
train_d= pd.concat([train_d, col4], axis=1)


col5 = pd.get_dummies(train_d.LotShape, prefix='LotShape')
del train_d['LotShape']
train_d= pd.concat([train_d, col5], axis=1)


col6 = pd.get_dummies(train_d.LandContour, prefix='LandContour')
del train_d['LandContour']
train_d= pd.concat([train_d, col6], axis=1)


col7 = pd.get_dummies(train_d.Utilities, prefix='Utilities')
del train_d['Utilities']
train_d= pd.concat([train_d, col7], axis=1)


col8 = pd.get_dummies(train_d.LotConfig, prefix='LotConfig')
del train_d['LotConfig']
train_d= pd.concat([train_d, col8], axis=1)

col9 = pd.get_dummies(train_d.LandSlope, prefix='LandSlope')
del train_d['LandSlope']
train_d= pd.concat([train_d, col9], axis=1)

col10 = pd.get_dummies(train_d.Neighborhood, prefix='Neighborhood')
del train_d['Neighborhood']
train_d= pd.concat([train_d, col10], axis=1)

col11 = pd.get_dummies(train_d.Condition1, prefix='Condition1')
del train_d['Condition1']
train_d= pd.concat([train_d, col11], axis=1)

col12 = pd.get_dummies(train_d.Condition2, prefix='Condition2')
del train_d['Condition2']
train_d= pd.concat([train_d, col12], axis=1)

col13 = pd.get_dummies(train_d.BldgType, prefix='BldgType')
del train_d['BldgType']
train_d= pd.concat([train_d, col13], axis=1)

col14 = pd.get_dummies(train_d.HouseStyle, prefix='HouseStyle')
del train_d['HouseStyle']
train_d= pd.concat([train_d, col14], axis=1)

col15 = pd.get_dummies(train_d.RoofStyle, prefix='RoofStyle')
del train_d['RoofStyle']
train_d= pd.concat([train_d, col15], axis=1)

col16 = pd.get_dummies(train_d.RoofMatl, prefix='RoofMatl')
del train_d['RoofMatl']
train_d= pd.concat([train_d, col16], axis=1)

col17 = pd.get_dummies(train_d.Exterior1st, prefix='Exterior1st')
del train_d['Exterior1st']
train_d= pd.concat([train_d, col17], axis=1)

col18 = pd.get_dummies(train_d.Exterior2nd, prefix='Exterior2nd')
del train_d['Exterior2nd']
train_d= pd.concat([train_d, col18], axis=1)


col19 = pd.get_dummies(train_d.MasVnrType, prefix='MasVnrType')
del train_d['MasVnrType']
train_d= pd.concat([train_d, col19], axis=1)

col20 = pd.get_dummies(train_d.Foundation, prefix='Foundation')
del train_d['Foundation']
train_d= pd.concat([train_d, col20], axis=1)

col21 = pd.get_dummies(train_d.Heating, prefix='Heating')
del train_d['Heating']
train_d= pd.concat([train_d, col21], axis=1)




col22 = pd.get_dummies(train_d.Functional, prefix='Functional')
del train_d['Functional']
train_d= pd.concat([train_d, col22], axis=1)

col23 = pd.get_dummies(train_d.GarageType, prefix='GarageType')
del train_d['GarageType']
train_d= pd.concat([train_d, col23], axis=1)

col24 = pd.get_dummies(train_d.GarageFinish, prefix='GarageFinish')
del train_d['GarageFinish']
train_d= pd.concat([train_d, col24], axis=1)

col25 = pd.get_dummies(train_d.PavedDrive, prefix='PavedDrive')
del train_d['PavedDrive']
train_d= pd.concat([train_d, col25], axis=1)

col26 = pd.get_dummies(train_d.SaleType, prefix='SaleType')
del train_d['SaleType']
train_d= pd.concat([train_d, col26], axis=1)

col27 = pd.get_dummies(train_d.SaleCondition, prefix='SaleCondition')
del train_d['SaleCondition']
train_d= pd.concat([train_d, col27], axis=1)

col28 = pd.get_dummies(train_d.Electrical, prefix='Electrical')
del train_d['Electrical']
train_d= pd.concat([train_d, col28], axis=1)

col29 = pd.get_dummies(train_d.Fence, prefix='Fence')
del train_d['Fence']
train_d= pd.concat([train_d, col29], axis=1)



# data transformation for testing dataset


#one hot vector encoding

col1 = pd.get_dummies(test_d.MSSubClass, prefix='MSSubClass')
del test_d['MSSubClass']
test_d= pd.concat([test_d, col1], axis=1)

col2 = pd.get_dummies(test_d.MSZoning, prefix='MSZoning')
del test_d['MSZoning']
test_d= pd.concat([test_d, col2], axis=1)


col3 = pd.get_dummies(test_d.Street, prefix='Street')
del test_d['Street']
test_d= pd.concat([test_d, col3], axis=1)


col4 = pd.get_dummies(test_d.Alley, prefix='Alley')
del test_d['Alley']
test_d= pd.concat([test_d, col4], axis=1)


col5 = pd.get_dummies(test_d.LotShape, prefix='LotShape')
del test_d['LotShape']
test_d= pd.concat([test_d, col5], axis=1)


col6 = pd.get_dummies(test_d.LandContour, prefix='LandContour')
del test_d['LandContour']
test_d= pd.concat([test_d, col6], axis=1)


col7 = pd.get_dummies(test_d.Utilities, prefix='Utilities')
del test_d['Utilities']
test_d= pd.concat([test_d, col7], axis=1)


col8 = pd.get_dummies(test_d.LotConfig, prefix='LotConfig')
del test_d['LotConfig']
test_d= pd.concat([test_d, col8], axis=1)

col9 = pd.get_dummies(test_d.LandSlope, prefix='LandSlope')
del test_d['LandSlope']
test_d= pd.concat([test_d, col9], axis=1)

col10 = pd.get_dummies(test_d.Neighborhood, prefix='Neighborhood')
del test_d['Neighborhood']
test_d= pd.concat([test_d, col10], axis=1)

col11 = pd.get_dummies(test_d.Condition1, prefix='Condition1')
del test_d['Condition1']
test_d= pd.concat([test_d, col11], axis=1)

col12 = pd.get_dummies(test_d.Condition2, prefix='Condition2')
del test_d['Condition2']
test_d= pd.concat([test_d, col12], axis=1)

col13 = pd.get_dummies(test_d.BldgType, prefix='BldgType')
del test_d['BldgType']
test_d= pd.concat([test_d, col13], axis=1)

col14 = pd.get_dummies(test_d.HouseStyle, prefix='HouseStyle')
del test_d['HouseStyle']
test_d= pd.concat([test_d, col14], axis=1)

col15 = pd.get_dummies(test_d.RoofStyle, prefix='RoofStyle')
del test_d['RoofStyle']
test_d= pd.concat([test_d, col15], axis=1)

col16 = pd.get_dummies(test_d.RoofMatl, prefix='RoofMatl')
del test_d['RoofMatl']
test_d= pd.concat([test_d, col16], axis=1)

col17 = pd.get_dummies(test_d.Exterior1st, prefix='Exterior1st')
del test_d['Exterior1st']
test_d= pd.concat([test_d, col17], axis=1)

col18 = pd.get_dummies(test_d.Exterior2nd, prefix='Exterior2nd')
del test_d['Exterior2nd']
test_d= pd.concat([test_d, col18], axis=1)


col19 = pd.get_dummies(test_d.MasVnrType, prefix='MasVnrType')
del test_d['MasVnrType']
test_d= pd.concat([test_d, col19], axis=1)

col20 = pd.get_dummies(test_d.Foundation, prefix='Foundation')
del test_d['Foundation']
test_d= pd.concat([test_d, col20], axis=1)

col21 = pd.get_dummies(test_d.Heating, prefix='Heating')
del test_d['Heating']
test_d= pd.concat([test_d, col21], axis=1)




col22 = pd.get_dummies(test_d.Functional, prefix='Functional')
del test_d['Functional']
test_d= pd.concat([test_d, col22], axis=1)

col23 = pd.get_dummies(test_d.GarageType, prefix='GarageType')
del test_d['GarageType']
test_d= pd.concat([test_d, col23], axis=1)

col24 = pd.get_dummies(test_d.GarageFinish, prefix='GarageFinish')
del test_d['GarageFinish']
test_d= pd.concat([test_d, col24], axis=1)

col25 = pd.get_dummies(test_d.PavedDrive, prefix='PavedDrive')
del test_d['PavedDrive']
test_d= pd.concat([test_d, col25], axis=1)

col26 = pd.get_dummies(test_d.SaleType, prefix='SaleType')
del test_d['SaleType']
test_d= pd.concat([test_d, col26], axis=1)

col27 = pd.get_dummies(test_d.SaleCondition, prefix='SaleCondition')
del test_d['SaleCondition']
test_d= pd.concat([test_d, col27], axis=1)

col28 = pd.get_dummies(test_d.Electrical, prefix='Electrical')
del test_d['Electrical']
test_d= pd.concat([test_d, col28], axis=1)

col29 = pd.get_dummies(test_d.Fence, prefix='Fence')
del test_d['Fence']
test_d= pd.concat([test_d, col29], axis=1)




'''xtra_feature=['Condition2_RRNn',
'HouseStyle_2.5Fin',
'RoofMatl_Membran',
'RoofMatl_Roll',
'Exterior1st_ImStucc',
'Exterior1st_Stone',
'Exterior2nd_Other',
'Heating_OthW',
'Electrical_Mix']

train_d.drop(xtra_feature,axis=1,inplace=True)'''
# ordinal features encoding training dataframe

train_d['CentralAir'] = train_d['CentralAir'].map({'Y':1, 'N':0})
train_d['ExterQual'] = train_d['ExterQual'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'Po':0})
train_d['ExterCond'] = train_d['ExterCond'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'Po':0})
train_d['BsmtQual'] = train_d['BsmtQual'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
train_d['BsmtCond'] = train_d['BsmtCond'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
train_d['BsmtExposure'] = train_d['BsmtExposure'].map({'Gd':4, 'Av':3,'Mn':2,'No':1,'No Basement':0})
train_d['BsmtFinType1'] = train_d['BsmtFinType1'].map({'GLQ':6, 'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'No':0})

train_d['BsmtFinType2'] = train_d['BsmtFinType2'].map({'GLQ':6, 'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'No':0})
train_d['HeatingQC'] = train_d['HeatingQC'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'Po':0})
train_d['KitchenQual'] = train_d['KitchenQual'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'Po':0})
train_d['FireplaceQu'] = train_d['FireplaceQu'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
train_d['GarageQual'] = train_d['GarageQual'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
train_d['GarageCond'] = train_d['GarageCond'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
train_d['PoolQC'] = train_d['PoolQC'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'No':0})

# ordinal features encoding testing dataframe

test_d['CentralAir'] = test_d['CentralAir'].map({'Y':1, 'N':0})
test_d['ExterQual'] = test_d['ExterQual'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'Po':0})
test_d['ExterCond'] = test_d['ExterCond'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'Po':0})
test_d['BsmtQual'] = test_d['BsmtQual'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
test_d['BsmtCond'] = test_d['BsmtCond'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
test_d['BsmtExposure'] = test_d['BsmtExposure'].map({'Gd':4, 'Av':3,'Mn':2,'No':1,'No Basement':0})
test_d['BsmtFinType1'] = test_d['BsmtFinType1'].map({'GLQ':6, 'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'No':0})

test_d['BsmtFinType2'] = test_d['BsmtFinType2'].map({'GLQ':6, 'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'No':0})
test_d['HeatingQC'] = test_d['HeatingQC'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'Po':0})
test_d['KitchenQual'] = test_d['KitchenQual'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'Po':0})
test_d['FireplaceQu'] = test_d['FireplaceQu'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
test_d['GarageQual'] = test_d['GarageQual'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
test_d['GarageCond'] = test_d['GarageCond'].map({'Ex':5, 'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})
test_d['PoolQC'] = test_d['PoolQC'].map({'Ex':4, 'Gd':3,'TA':2,'Fa':1,'No':0})
train_d.count()
train_d.isnull().values.any()
train_d.count()
#check columns with null value
test_d.loc[:, test_d.isna().any()]
#Comparison between mean, median and mode to proceed with imputation 

print(test_d[["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","GarageArea"]].mean())
print(test_d[["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","GarageArea"]].median())
print(test_d[["BsmtFullBath","BsmtHalfBath","KitchenQual","GarageCars"]].mode())
#imputation of scalar/nominal values in test dataset

test_d['BsmtFinSF1'] = test_d['BsmtFinSF1'].fillna(test_d['BsmtFinSF1'].mean())
test_d['BsmtFinSF2'] = test_d['BsmtFinSF2'].fillna(test_d['BsmtFinSF2'].mean())
test_d['BsmtUnfSF'] = test_d['BsmtUnfSF'].fillna(test_d['BsmtUnfSF'].mean())
test_d['TotalBsmtSF'] = test_d['TotalBsmtSF'].fillna(test_d['TotalBsmtSF'].mean())
test_d['GarageArea'] = test_d['GarageArea'].fillna(test_d['GarageArea'].median())

#observed that mode doesn't work with fillna


test_d['BsmtFullBath'] = test_d['BsmtFullBath'].fillna(0.0)
test_d['BsmtHalfBath'] = test_d['BsmtHalfBath'].fillna(0.0)
test_d['KitchenQual'] = test_d['KitchenQual'].fillna(2.0)
test_d['GarageCars'] = test_d['GarageCars'].fillna(2.0)
test_d.isnull().values.any()
#test_d['BsmtFullBath'].value_counts()
# concise syntax to check for null values. keep it in code
null_columns=test_d.columns[test_d.isnull().any()]
test_d[null_columns].isnull().sum()
# code taken from Baris Camli's Kernel
from scipy import stats 
from scipy.stats import norm

sns.distplot(train_d['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_d['SalePrice'], plot=plt)
#Here our curve is right skewed so log transformation is one efficient way for scaling of SalePrice

train_d["SalePrice"] = np.log(train_d["SalePrice"])
# check if transformed data reflect properly
#train_d.to_csv(r'~/Documents/MachinelearningRepository/KaggleHousePriceProject/TransformedData.csv', index = False)
y=train_d['SalePrice']

x=train_d.drop(['SalePrice'],axis=1)


X=test_d
# dropping xtra features from train dataset to implement Lasso on Test:
for i in x.columns:   
    if not i in X:
        print(i)
cnt=0
for i in x.columns:
    if  i in X:
        cnt +=1
print(cnt)        
xtra_feature=['Utilities_NoSeWa',
'Condition2_RRAe',
'Condition2_RRAn',
'Condition2_RRNn',
'HouseStyle_2.5Fin',
'RoofMatl_Membran',
'RoofMatl_Metal',
'RoofMatl_Roll',
'Exterior1st_ImStucc',
'Exterior1st_Stone',
'Exterior2nd_Other',
'Heating_Floor',
'Heating_OthW',
'Electrical_Mix']

x.drop(xtra_feature,axis=1,inplace=True)
for j in X.columns:
    if  not j in x:
        print(j)
X.drop(columns=['MSSubClass_150'],axis=1,inplace=True)
x.count()

X.count()
# split tainining dataset into train and test


from sklearn.model_selection import train_test_split

X_train_org, X_test_org, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.25)

#Check the number of records in training and test partitions
print("X_train unscaled : " + str(X_train_org.shape))
print("X_test unscaled: " + str(X_test_org.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

from sklearn.linear_model import Lasso
x_range = [0.01, 0.1, 1, 10, 100]
train_score_list = []
test_score_list = []

for alpha in x_range: 
    lasso = Lasso(alpha)
    lasso.fit(X_train_org,y_train)
    train_score_list.append(lasso.score(X_train_org,y_train))
    test_score_list.append(lasso.score(X_test_org, y_test))
print(train_score_list)
print(test_score_list)
plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_range, test_score_list, c = 'b', label = 'Test Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel(r'$\alpha$')
# Grid Search for Algorithm Tuning
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# prepare a range of alpha values to test
param_grid = {
    'alpha': [100,10,2,1,0.5,0.1,0.01,0.001],
    'max_iter': [10e5]
    
}
# create and fit a lasso regression model, testing each alpha
model = Lasso()
grid = GridSearchCV(model,param_grid)
grid.fit(X_train_org, y_train)
yPredictions3 =grid.predict(X_test_org)


# summarize the results of the grid search
print ("Train Score:",grid.score(X_train_org, y_train))
print ("Test Score :",grid.score(X_test_org, y_test))
print("Best_Value_For_Alpha:",grid.best_params_)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error

lasso = Lasso(alpha = 1, max_iter = 1000000.0)

lasso.fit(X_train_org, y_train)
y_pred_lasso=lasso.predict(X_test_org)

# Cross Validation
cv_scores = cross_val_score(lasso, X_train_org, y_train,cv = 5)
cv_test_scores = cross_val_score(lasso, X_test_org, y_test,cv = 5)
print('Cross-validation training scores (5-fold):', cv_scores)
print('Cross-validation testing scores (5-fold):', cv_test_scores)
print('Mean cross-validation training score (5-fold): {:.4f}'.format(np.mean(cv_scores)))
print('Mean cross-validation testing score (5-fold): {:.4f}'.format(np.mean(cv_test_scores)))
print('RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(y_test,y_pred_lasso))))
Y_test_pred=lasso.predict(X)
print(Y_test_pred)
len(Y_test_pred)
# code taken from Baris Camli to take exponents of our results
Predicted_prices = np.exp(Y_test_pred) 
Predicted_prices
result = pd.DataFrame(data = test_IdBackup["Id"])
result["SalePrice"]= Predicted_prices
