# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
####import train and test dataset
train_input = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_input = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_input.info()
train_input.sample(10)
###visulization of the features vs saleprice
fig = plt.figure(figsize = (18,12))
plt.subplot2grid((2,2),(0,0))
plt.scatter(train_input['LotArea'], train_input['SalePrice'],alpha=0.5 )
plt.xlabel('LotArea')
plt.ylabel('SalePrice')

plt.subplot2grid((2,2),(0,1))
plt.scatter(train_input['YearBuilt'], train_input['SalePrice'],alpha=0.5 )
plt.xlabel('YearBuilt')
plt.ylabel('SalePrice')

plt.subplot2grid((2,2),(1,0))
plt.scatter(train_input['TotalBsmtSF'], train_input['SalePrice'],alpha=0.5 )
plt.xlabel('TotalBsmtSF')
plt.ylabel('counts')

plt.subplot2grid((2,2),(1,1))
train_input['BedroomAbvGr'].value_counts(normalize=True).plot(kind='bar', alpha=0.5)
plt.xlabel('BedroomAbvGr')
plt.ylabel('counts')
plt.show()




##Bedroom vs price
'''
fig = plt.figure(figsize = (18,12))
fig1 = plt.subplot2grid((2,2),(0,0))
fig2 = plt.subplot2grid((2,2),(0,1))
fig3 = plt.subplot2grid((2,2),(1,0))
fig4 = plt.subplot2grid((2,2),(1,1))
'''

bedvp = train_input[['SalePrice', 'BedroomAbvGr']].groupby('BedroomAbvGr', as_index=True).mean().plot(kind='bar', alpha=0.5)
#train_input.BedroomAbvGr[train_input['SalePrice'].value_counts].plot(kind='bar', alpha=0.5)
print(bedvp)
#plt.xlabel('BedroomAbvGr')
#plt.ylabel('SalePrice')
#print(train_input.SalePrice[train_input.BedroomAbvGr == 0].mean())   $221493.1666
#print(train_input.SalePrice)
#print(train_input['SalePrice'][train_input.BedroomAbvGr == 0])

##Fullbath vs price
fbathvp = train_input[['SalePrice', 'FullBath']].groupby('FullBath', as_index=True).mean().plot(kind='bar', alpha=0.5)
#train_input.BedroomAbvGr[train_input['SalePrice'].value_counts].plot(kind='bar', alpha=0.5)
print(fbathvp)

##GarageCars vs price
grgvp = train_input[['SalePrice', 'GarageCars']].groupby('GarageCars', as_index=True).mean().plot(kind='bar', alpha=0.5)

##Roofstyle vs price
rfstylvp = train_input[['SalePrice', 'RoofStyle']].groupby('RoofStyle', as_index=True).mean().plot(kind='bar', alpha=0.5)
print(train_input[['SalePrice', 'RoofStyle']].groupby('RoofStyle').mean())

print(train_input[['SalePrice', 'RoofStyle']].groupby('RoofStyle').mean())
print(train_input['SaleType'].unique())
print(train_input[['SalePrice', 'SaleType']].groupby('SaleType').mean())
##feature engineering step 1,select the features that may be useful by drop any columns contains Nulls
train_cp = train_input.copy(deep=True)
test_cp = test_input.copy(deep=True)
print('test_cp size:', test_cp.shape, 'train_cp size:', train_cp.shape)
#train_cp.isna().any()
train_fullcol = train_cp.dropna(axis = 1).reset_index(drop=True)
test_fullcol = test_cp.dropna(axis = 1).reset_index(drop=True)
print('test_fullcol size:', test_fullcol.shape, 'train_fullcol size:', train_fullcol.shape)
train_fullcol.sample(10)
##feature engineering step 2,select the features are int
train_intcol = train_fullcol.select_dtypes(exclude=['object'])
test_intcol = test_fullcol.select_dtypes(exclude=['object'])
#train_intcol.info()
#test_intcol.info()
##made sure both train_intcol and test_intcol has no Nulls, and are int64 dtypes
train_intcol.sample(5)
test_intcol.sample(5)
print('test_intcol size:', test_intcol.shape, 'train_intcol size:', train_intcol.shape)
train_intcol.describe()
##found a lot of 0 in BsmtFinSF2, EnclosedPorch, MiscVal, etc.
##will use all of these features to create a baseline model first, and then come back to tackle these features with a lot of 0s
train_intcol.filter(regex='SF').sample(15)
#these features have a lot of 0s: BsmtFinSF2, 2ndFlrSF, LowQualFinSF
train_intcol.filter(regex='Porch').sample(15)
#these features have a lot of 0s: EnclosedPorch, 3SsnPorch, ScreenPorch
test_feature_left = test_intcol.columns.values
train_cp[test_feature_left].sample(3)
#print(test_feature_left)
train_feature_left = train_intcol.columns.values
#print(train_feature_left.tolist())
diff_feature = set(train_feature_left.tolist()) - set(test_feature_left.tolist())
print(diff_feature)    ##use set to calculate the features present in A but not in B

train_cp[diff_feature].sample(15)
train_input[diff_feature].info()
feature_names = train_intcol.columns.drop(['SalePrice'])    ###index object operation refer to: https://pandas.pydata.org/pandas-docs/stable/reference/indexing.html
#n = np.delete(train_intcol.columns, 'SalePrice')
#test_input[feature_names].info()
diff_feature2 = diff_feature - set(['SalePrice'])
#test_input[feature_names].notnull().sum()-1459<0

for i in diff_feature2:
    
    test_cp[i].fillna(test_cp[i].dropna().median(), inplace=True)   ##use median to fill nan
    test_cp[i] = test_cp[i].astype('int64')

#test_cp[diff_feature2].info()
#train_cp[diff_feature2].info()
#test_cp[diff_feature2].sample(5)

test_intcol = test_cp[feature_names]
test_intcol.info()
test_cp[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']]
##feature engineering t4: to create/combine some features first
train_mod = train_intcol.copy()
test_mod = test_intcol.copy()
data = [train_mod, test_mod]
for each in data:
    each['TotalBath'] = each['FullBath'] + each['BsmtFullBath'] + each['HalfBath']*0.5 + each['BsmtHalfBath']*0.5   ##combining the bath#
    each.loc[each['BsmtUnfSF'] == 0, 'Basefinish'] = 0      ##simplify the basement to be finished/unfinished
    each.loc[each['BsmtUnfSF'] != 0, 'Basefinish'] = 1
    each['TotalSize'] = each['1stFlrSF'] + each['2ndFlrSF']
    temp = each['TotalSize'].values.reshape(-1,1)
    scaler = StandardScaler().fit(temp)
    each['Sizescl'] = scaler.transform(temp)
    each.loc[each['PoolArea'] == 0, 'Pool'] = 0     #1453/1460 has no pools, so just simplify it
    each.loc[each['PoolArea'] != 0, 'Pool'] = 1 
    '''
    train_input[['ScreenPorch', 'SalePrice']].groupby('ScreenPorch').count()   #counts: 1344 vs 75
    train_input[['EnclosedPorch', 'SalePrice']].groupby('EnclosedPorch').count()  #counts: 1252 vs 119
    train_input[['3SsnPorch', 'SalePrice']].groupby('3SsnPorch').count() #0 counts: 1436 vs 19
    train_input[['OpenPorchSF', 'SalePrice']].groupby('OpenPorchSF').count()   #0 counts: 656 vs 201
    '''
    each['PorchType'] = 0
    each.loc[each['ScreenPorch'] != 0, 'PorchType'] = 1
    each.loc[each['EnclosedPorch'] != 0, 'PorchType'] = 2
    each.loc[each['3SsnPorch'] != 0, 'PorchType'] = 3
    each.loc[each['OpenPorchSF'] != 0, 'PorchType'] = 4
    #train_input[['LowQualFinSF', 'SalePrice']].groupby('LowQualFinSF').count().shape  only 23 with LowQualFinSF
    #each['LowQualFin'] = 0
    #each.loc[each['LowQualFinSF'] != 0, 'LowQualFin'] = 1
    each['Sold-Built'] = each['YrSold'] - each['YearBuilt']
    #each['Remod-Built'] = each['YearRemodAdd'] - each['YearBuilt']
    
    
    each.drop(['Id', 'TotalSize', 'MiscVal', 'ScreenPorch', 'EnclosedPorch', '3SsnPorch', 'OpenPorchSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'FullBath', 'HalfBath', '1stFlrSF', '2ndFlrSF', '1stFlrSF', '2ndFlrSF', 'BsmtHalfBath', 'BsmtFullBath', 'PoolArea' ], axis = 1, inplace=True)     

train_mod.info()


train_input[['MiscVal', 'SalePrice']].groupby('MiscVal').mean().plot() #tried to drop this feature, and the score improved a bit
train_mod.sample(10)
#train_mod.describe()
rest_feature = set(train_cp.columns.values) - set(train_intcol.columns.values)
rest_fpd = pd.DataFrame([rest_feature]).T
train_input[rest_feature].sample(5)

train_input[rest_feature].info()
train_input['RoofMatl'].value_counts()
train_input['SalePrice'].mean()
train_input['MasVnrType'].unique()
a = train_input[['RoofMatl','SalePrice']].groupby('RoofMatl').count()
b = train_input[['RoofMatl','SalePrice']].groupby('RoofMatl').mean()
c = train_input[['RoofMatl','SalePrice']].groupby('RoofMatl').std()/train_input['SalePrice'].mean()
fig, ax= plt.subplots(1,1,figsize=(12,5))
c.plot.bar(ax=ax, label = 'Count')
b.plot(ax=ax, secondary_y=True, color='r', label = 'SalePrice')
#print(a.values[4][0])
#print(a.index.tolist()[2])

for ind, each in enumerate(ax.patches):
    height = each.get_height()
    x = each.get_x()  + each.get_width()/4
    ax.annotate(a.values[ind][0], (x, height*1.005))
#c.plot(ax=ax, secondary_y=True, color='g')
#train_input[['Alley','SalePrice']].groupby('Alley').std()/train_input[['Alley','SalePrice']].groupby('Alley').mean()
###visualize to QC the rest feature
for i in rest_feature:
    #print(train_input[[i,'SalePrice']].groupby(i).count())
    a = train_input[[i,'SalePrice']].groupby(i).count()
    b = train_input[[i,'SalePrice']].groupby(i).mean()
    c = train_input[[i,'SalePrice']].groupby(i).std()/train_input['SalePrice'].mean()
    fig, ax= plt.subplots(1,1,figsize=(12,5))
    c.plot.bar(ax=ax)
    b.plot(ax=ax, secondary_y=True, color='r')
    for ind, each in enumerate(ax.patches):
        height = each.get_height()
        x = each.get_x()  + each.get_width()/4
        ax.annotate(a.values[ind][0], (x, height*1.005))

orig24f = train_mod.columns.drop(['SalePrice']).values.tolist()
print(orig24f)
print(train_mod[['Sold-Built','SalePrice']].groupby('Sold-Built').mean())
###QC the original 24 features:

for i in orig24f:
    #print(train_input[[i,'SalePrice']].groupby(i).count())
    a = train_mod[[i,'SalePrice']].groupby(i).count()
    b = train_mod[[i,'SalePrice']].groupby(i).mean()
    c = train_mod[[i,'SalePrice']].groupby(i).std()/train_input['SalePrice'].mean()
    fig, ax= plt.subplots(1,1,figsize=(12,5))
    c.plot.bar(ax=ax)
    b.plot(ax=ax, secondary_y=True, color='r')
    for ind, each in enumerate(ax.patches):
        height = each.get_height()
        x = each.get_x()  + each.get_width()/4
        ax.annotate(a.values[ind][0], (x, height*1.005))
train_mod2 = train_input[selectedf].copy()
train_mod2['GarageFinish'].fillna('Other', inplace=True)
###manually picked 25 features as the first batch to convert and complete
#selectedf = ['GarageFinish', 'RoofMatl', 'KitchenQual', 'BsmtQual', 'Foundation', 'ExterQual', 'ExterCond', 'Exterior2nd', 'HeatingQC', 'RoofStyle', 'LandSlope', 'CentralAir', 'BsmtExposure', 'PavedDrive', 'BsmtCond', 'MasVnrType', 'SaleCondition', 'Functional']
selectedf = ['GarageType', 'BsmtExposure', 'Foundation', 'BldgType', 'HeatingQC', 'GarageFinish', 'CentralAir', 'KitchenQual', 'Exterior1st', 'Electrical', 'Condition1', 'Heating', 'BsmtQual', 'SaleCondition', 'LandSlope', 'PavedDrive', 'GarageCond', 'MasVnrType', 'ExterQual', 'BsmtCond', 
            'BsmtFinType2', 'BsmtFinType1', 'HouseStyle', 'SaleType', 'MSZoning']
#train_input[selectedf].info()
#test_input[selectedf].info()


##step 1, completing the data using method 1,create a new  category; 2, interpolate with the most popular one. 
##step 2, converting the object to int by using labelencoder
train_mod2 = train_input[selectedf].copy()
test_mod2 = test_input[selectedf].copy()
data2 = [train_mod2, test_mod2]
label = LabelEncoder()
for each in data2:
    for i in selectedf:
        each[i].fillna('Other', inplace=True)  #1,create a new  category
        #each[i].fillna(each[i].mode()[0], inplace=True)    #2,  interpolate with the most popular one. 
        each[i] = label.fit_transform(each[i])
    
#train_mod2['GarageFinish'] = train_mod2['GarageFinish'].apply(pd.to_numeric, errors='coerce') 
#train_mod2['GarageFinish'].fillna('Other', inplace=True)
#train_mod2['GarageFinish'].fillna(train_mod2['GarageFinish'].dropna.median(), inplace=True)  ##median doesn't work with Obj
#train_mod2['GarageFinish'] = label.fit_transform(train_mod2['GarageFinish'])




print(test_mod2.sample(5))
train_mod2.info()
##combinint the additional 25 features to original 24 features, totaled 42 features
train_comb = pd.concat([train_mod, train_mod2],axis=1)
test_comb = pd.concat([test_mod, test_mod2],axis=1)
train_comb.info()
###hyperparameter tuning:
from sklearn import model_selection
feature_names = test_comb.columns.values
target = train_comb['SalePrice'].values
features = train_comb[feature_names].values

param_grid = {'n_estimators': [50, 100, 200, 500, 1000], 'max_depth': [4, 6, 8, 10], 'max_features': [0.6, 0.8, 1], 'random_state': [10]}
rfclf = RandomForestRegressor()
model_tune = model_selection.GridSearchCV(rfclf, param_grid = param_grid, cv=20)
model_tune.fit(features,target)
print(model_tune.best_params_)
##throw these features into a baseline model to see how the result turns out to be:
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

feature_names = test_comb.columns.values
target = train_comb['SalePrice'].values
features = train_comb[feature_names].values

##Random Forest
rfclf = RandomForestRegressor(n_estimators = 500, max_depth = 14, max_features = 0.6, random_state = 10)
rfc = rfclf.fit(features, target)
#print(rfclf.score(features, target))
#print(cross_val_score(rfclf, features, target, scoring='explained_variance', cv=10).mean())
#using 34 features from train_intcol get:0.98081  cv:0.862532       baseline test score:0.15796
#using 26 features from test_intcol get:0.978712  cv:0.846705       
#using 34 features from train_intcol + fillna test_intcol get:0.9808 cv:0.8625        test score:0.15065
#using 28 features after combine/discard get:0.9776 cv:0.8516         test score:0.15075
#0611 using25 features after combine/discard get:0.9791 cv:0.8527    test score:0.15024
#using 24 features(no MiscVal), get:0.9792 cv:0.8536     test score:0.15057
#using 24 features scale TotalSize, get:0.9787 cv:0.8520   test score:0.15020
#using 25 features scale totalsize, get:0.9783 cv:0.8519   test score:0.15025
#0612using 23 features: no MiscVal no Id, get:0.979779 cv:0.851552
#0612using 24 features: no MiscVal no Id, add Sold - Build: get:0.98018 cv:0.85350  test score:0.14826
#0613 on top of the previous test, add 18 more features(fill with new category), get:0.98072 cv:0.85917  test score:0.14632
#0613 on top of the previous test, add 18 more features(fill with new category) + hp tuning, get:0.97496 cv:0.86611  test score:0.14610
#0613 on top of the previous test, add 18 more features(fill with most frequent) + hp tuning, get:0.97480 cv:0.86586  test score:0.14617
#0613 on top of the previous test, add 25 more features(fill with new category) + (old)hp tuning, get:0.97520 cv:0.86479  test score:0.14814
#0616 on top of the previous test, add 25 more features(fill with new category) + (old)hp tuning +maxdepth 14, get:0.98082 cv:0.86851 
#0616 on top of the previous test, add 25 more features(fill with new category) + (old)hp tuning +maxdepth 18, get:0.98202 cv:0.86754 test score:0.14683
#0616 on top of the previous test, add 25 more features(fill with new category) + (old)hp tuning +maxdepth 22, get:0.98143 cv:0.86664
###baseline model submission
features = test_comb[feature_names].values
predictions = rfclf.predict(features)

output = pd.DataFrame({'Id': test_intcol.Id, 'SalePrice': predictions})
output.to_csv('houseprice_0616_49feat_newhpt.csv', index=False)
print('Your submission was successfully saved!')



