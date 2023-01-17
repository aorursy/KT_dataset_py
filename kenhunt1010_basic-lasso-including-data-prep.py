import pandas as pd

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

import numpy as np

from sklearn.preprocessing import LabelEncoder



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass 

warnings.warn = ignore_warn #ignore warning from sklearn and seaborn



#pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) #Limiting floats output to 2 decimal points

# Path of the file to read. 

train_set = '../input/train.csv'

train = pd.read_csv(train_set)



train.head()

#Save the Primary Key IDcolumn in case we need it later

train_ID = train['Id']



#Drop Primary Key - 

train.drop("Id", axis = 1, inplace = True)
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, square=True);

#scatterplot correlated variables.

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show()
#plot scatter to identify any outliers

var='GrLivArea'



fig, ax = plt.subplots()

ax.scatter(x = train[var], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=10)

plt.xlabel(var, fontsize=10)

plt.show()
#Delete outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Check the graph again

vari = 'GrLivArea'

fig, ax = plt.subplots()

ax.scatter(train[vari], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel(vari, fontsize=13)

plt.show()
#Plot the distribution of the Target column Y.

sns.distplot(train['SalePrice'])



#Sort out the distribution by using Log1p

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice'])
#Show % of nulls per column

train_na_percent = (train.isnull().sum() / len(train)).sort_values(ascending=False)

train_na_total = train.isnull().sum().sort_values(ascending=False)

missing_data = pd.concat([train_na_percent, train_na_total], axis=1, keys=['%', 'Total'])[:30]

missing_data.head()
#Graph it 

f, ax = plt.subplots(figsize=(6, 4))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['%'])

plt.xlabel('Feature')

plt.ylabel('Percent of missing values')

#Drop Columns with poor data



#train = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)





##FillNA with String

for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','MSSubClass', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    train[col] = train[col].fillna('None')

     

#FillNA with Median

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



#fillNA with Zero    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):

    train[col] = train[col].fillna(0)



#fillNA with Mode    

for col in ('MSZoning', 'Electrical', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType'):

    train[col] = train[col].fillna(train[col].mode()[0])



#Drop

train = train.drop(['Utilities'], axis=1)



#Functional replace NA with Typ as per data descritpion - thanks @Sergine

train["Functional"] = train["Functional"].fillna("Typ")





#Check remaining missing values if any 

train_na = (train.isnull().sum() / len(train)) * 100

train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame(train_na)

missing_data.head()
#MSSubClass needs to be str

train['MSSubClass'] = train['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

train['OverallCond'] = train['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

train['YrSold'] = train['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)
#label encoding for all of the categorical varibles that are stored as object - also added in the remaining "object" type features.





cols = ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold', 'MSZoning','LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 

        'Condition2',  'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 

        'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition')





# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(train[c].values)) 

    train[c] = lbl.transform(list(train[c].values))



#Check if there are any "object" types left. (used previously for randomForest)

train.info()
# Adding total sqfootage feature 

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
train.head()
# Split Data set into X and Y - X is the features, Y is the varible we want to predict.

X = train

X = train.drop(['SalePrice'], axis=1) #remove when using test data

y = train['SalePrice'] #remove when using test data



X.shape

# Select an alpha - Trail and Error or Cross Validation - will expand on this.

best_alpha = 0.00099



#Train the Model

regr = Lasso(alpha=best_alpha, max_iter=50000)

regr.fit(X, y)

#Predict using X as our parameters

lasso_pred = regr.predict(X)
print(lasso_pred)

#Add the targets and predictiions back into the data set also use expm1 to change them back to their original numbers



#Add the score to X

X['lassoScore'] = lasso_pred



#Add the Actual Score to X

X['lassoScoreAct']= np.expm1(X['lassoScore'])



#Create a column named Targets containing the original SalesPrice feature 

X['Targets'] = y

X.head()



#Graph Targets vs Prediction to see correllation

sns.scatterplot(x=X['lassoScore'], y=X['Targets'])

plt.xlabel('Target')

plt.ylabel('Prediction')

plt.show()
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

from math import sqrt





rf_val_sq_error2 = sqrt(mean_squared_error(X['Targets'], X['lassoScore'])) #correct

print(rf_val_sq_error2)







#for submission of test data set

#submission = X[['Id', 'SalePrice']]

#submission.to_csv('submission.csv', index=False)




