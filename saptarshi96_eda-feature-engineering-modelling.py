import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as st

from sklearn import ensemble, tree, linear_model

import missingno as msno



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



plt.style.use('classic')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print(train.head())

#print(train.columns.value_counts())

print(train.shape)

print(test.shape)
test_Id = test['Id']

train.drop('Id',axis=1,inplace=True)

test.drop('Id',axis=1,inplace=True)
sns.distplot(train['SalePrice'], bins = 30)
# Skew and kurt

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
#train['SalePrice']= np.log1p(train['SalePrice'])

#sns.distplot(train['SalePrice'], bins = 30)
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.columns
numeric_features.head()
date_feature = [feature for feature in numeric_features if 'Yr' in feature or 'Year' in feature]

print(date_feature)



for feature in date_feature:

  print(feature, train[feature].unique())
#plt.scatter(x = (train['YrSold'] - train['YearBuilt']), y = train['SalePrice'])



for feature in date_feature:

  if feature != 'YrSold':

    data = train.copy()

    plt.scatter(x = (data['YrSold'] - data[feature]), y = data['SalePrice'])

    plt.xlabel(feature)

    plt.ylabel('Sale Price')

    plt.show()

discrete_feature = [ feature for feature in numeric_features if len(train[feature].unique()) < 25 

                    and feature not in date_feature + ['Id']]



print(len(discrete_feature))
train[discrete_feature].head()
for feature in discrete_feature:

  data = train.copy(feature)

  data = data.groupby(feature)['SalePrice']

  data.median().plot.bar()

  plt.xlabel(feature)

  plt.ylabel('SalePrice')

  plt.title(feature)

  plt.show()
continious_feature = [ feature for feature in numeric_features if feature not in date_feature + discrete_feature + ['Id']]

print(len(continious_feature))
for feature in continious_feature:

  data = train.copy(feature)

  data[feature].hist(bins = 30)

  plt.xlabel(feature)

  plt.ylabel('Count')

  plt.title(feature)

  plt.show()
fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(14,10))

OverallQual_scatter_plot = pd.concat([train['SalePrice'],train['OverallQual']],axis = 1)

sns.regplot(x='OverallQual',y = 'SalePrice',data = OverallQual_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

TotalBsmtSF_scatter_plot = pd.concat([train['SalePrice'],train['TotalBsmtSF']],axis = 1)

sns.regplot(x='TotalBsmtSF',y = 'SalePrice',data = TotalBsmtSF_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

GrLivArea_scatter_plot = pd.concat([train['SalePrice'],train['GrLivArea']],axis = 1)

sns.regplot(x='GrLivArea',y = 'SalePrice',data = GrLivArea_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

GarageArea_scatter_plot = pd.concat([train['SalePrice'],train['GarageArea']],axis = 1)

sns.regplot(x='GarageArea',y = 'SalePrice',data = GarageArea_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

FullBath_scatter_plot = pd.concat([train['SalePrice'],train['FullBath']],axis = 1)

sns.regplot(x='FullBath',y = 'SalePrice',data = FullBath_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

YearBuilt_scatter_plot = pd.concat([train['SalePrice'],train['YearBuilt']],axis = 1)

sns.regplot(x='YearBuilt',y = 'SalePrice',data = YearBuilt_scatter_plot,scatter= True, fit_reg=True, ax=ax6)
YearRemodAdd_scatter_plot = pd.concat([train['SalePrice'],train['YearRemodAdd']],axis = 1)

YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd','SalePrice')
catagorical_features = train.select_dtypes(include = np.object)

catagorical_features.columns
sns.boxplot(x='OverallQual', y="SalePrice", data=train)
plt.figure(figsize=(20,5))

sns.boxplot(x='Neighborhood', y="SalePrice", data=train)

plt.xticks(rotation=90)
def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)





f = pd.melt(train, id_vars=['SalePrice'], value_vars=catagorical_features)#check the output using f.head()

g = sns.FacetGrid(f, col="variable",  col_wrap=5,sharex=False, sharey=False)

g = g.map(boxplot, "value", "SalePrice")

plt.xticks(rotation=90)

#sns.FacetGrid(data, row=None, col=None, hue=None, col_wrap=None, sharex=True, sharey=True)
plt.figure(figsize=(20,5))

sns.boxplot(x='GarageYrBlt', y="SalePrice", data=train)

plt.xticks(rotation=90)
plt.figure(figsize = (12,12))

sns.heatmap(train.corr())



correlation = train.corr()
k= 15

cols = correlation.nlargest(k,'SalePrice')['SalePrice'].index

print(cols)
plt.figure(figsize = (12,12))



sns.heatmap(train[cols].corr(), vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white")
train=train.drop(['GarageCars','1stFlrSF','TotRmsAbvGrd'],axis=1)

test=test.drop(['GarageCars','1stFlrSF','TotRmsAbvGrd'],axis=1)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
df = train

# Keep only the rows with at least 4 non-na values

df.dropna(axis =1 ,inplace = True, thresh = 300)



print('No. of Columns:' +str(len(df.columns)))



total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(missing_data.head(20))



#We dropped Four Columns



print('No of rows:' +str(len(df)))
test=test.drop(['PoolQC','MiscFeature','Alley','Fence'],axis=1)
# Will drop the rows only if all of the values in the row are missing

df.dropna(how = 'all',inplace = True)



print('No. of rows:' +str(len(df))) #No rows dropped as condition not satisfied
df.dropna(how='any', subset=['MasVnrArea', 'MasVnrType', 'Electrical'], inplace = True)

print('No. of rows:' +str(len(df)))



total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(missing_data.head(20))
catagorical_features = df.select_dtypes(include = np.object)

catagorical_features.columns
for feature in catagorical_features:

  df.fillna('0', inplace = True)

  test.fillna('0', inplace = True)





total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(missing_data.head(20))
numeric_features = df.select_dtypes(include=[np.number])
date_feature = [feature for feature in numeric_features if 'Yr' in feature or 'Year' in feature]



for feature in date_feature:

  df[feature].fillna(df[feature].mode()[0],inplace=True)
discrete_feature = [ feature for feature in numeric_features if len(train[feature].unique()) < 25 

                    and feature not in date_feature + ['Id']]



for feature in date_feature:

  df[feature].fillna(df[feature].mode()[0],inplace=True)
continious_feature = [ feature for feature in numeric_features if feature not in date_feature + discrete_feature + ['Id']]



for feature in continious_feature:

  df[feature].fillna(df[feature].mean(),inplace=True)

print('No. of rows:' +str(len(df)))

print('No. of columns:' +str(len(df.columns)))
train = df

print(train.shape)

print(test.shape)
num_train=train[['OverallCond','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','KitchenAbvGr',

'BedroomAbvGr','Fireplaces','MoSold','YrSold', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']]

num_test=test[['OverallCond','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','KitchenAbvGr',

'BedroomAbvGr','Fireplaces','MoSold','YrSold', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']]

print(num_train.shape)

print(num_test.shape)

numerical_categorical_feature=c = pd.concat((num_train,num_test),sort=False)

print(numerical_categorical_feature.shape)

numerical_categorical_feature=numerical_categorical_feature.astype('O')
numerical_categorical_feature = pd.get_dummies(numerical_categorical_feature)

num_train_dummy = numerical_categorical_feature[:1451]

num_test_dummy = numerical_categorical_feature[1451:]

print(num_train_dummy.shape)

print(num_test_dummy.shape)
train=train.drop(['OverallCond','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','KitchenAbvGr',

'BedroomAbvGr','Fireplaces','MoSold','YrSold', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'],axis=1)



test=test.drop(['OverallCond','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','KitchenAbvGr',

'BedroomAbvGr','Fireplaces','MoSold','YrSold', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'],axis=1)



print(train.shape)

print(test.shape)

#skew_features = train.apply(lambda x : x.skew()).sort_values(ascending = False)

#high_skew = skew_features[skew_features>0.5]

#high_skew = high_skew.index

#print('There are '+str(skew_index.shape[0]) +' highly skewed numerical features in training set')

#print(high_skew)
# Normalize skewed features

#train[high_skew] = np.log1p(train[high_skew])
#skew_features = test.apply(lambda x : x.skew()).sort_values(ascending = False)

#high_skew_test = skew_features[skew_features>0.5]

#high_skew_test = high_skew_test.index

#print('There are '+str(high_skew_test.shape[0]) +' highly skewed numerical features in test set')

#print(high_skew_test)
#test[high_skew_test] = np.log1p(test[high_skew_test])
final_train = train.merge(num_train_dummy,left_index=True,right_index=True)

final_test = test.merge(num_test_dummy,left_index=True,right_index=True)



print(final_train.shape)

print(final_test.shape)
from sklearn.model_selection import train_test_split



X = final_train.drop('SalePrice',axis=1)

y = final_train['SalePrice']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor()

dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
predictions = dtree.predict(final_test)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 501, random_state = 0)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
predictions = rf.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
predictions = rf.predict(final_test)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedKFold

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()

# define the evaluation procedure

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate the model

n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# report performance

print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
model.fit(X, y)
predictions = model.predict(final_test)
sub = pd.DataFrame()

sub['Id'] = test_Id

sub['SalePrice'] = predictions

sub.to_csv('submission.csv',index=False)