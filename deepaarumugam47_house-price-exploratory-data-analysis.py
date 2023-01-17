# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as st


from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()
test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.head()
train.info()
train.describe()
train.shape , test.shape
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns
numeric_features.head()
# list of variables that contain year information
year_feature = [feature for feature in numeric_features if 'Yr' in feature or 'Year' in feature]

year_feature
# Let us explore the contents of temporal  variables
for feature in year_feature:
    print(feature, train[feature].unique())
for feature in year_feature:
    if feature!='YrSold':
        data=train.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel( feature)
        plt.ylabel('SalePrice')
        plt.show()
discrete_feature=[feature for feature in numeric_features if len(train[feature].unique())<25 and feature not in year_feature + ['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
train[discrete_feature].head()
for feature in discrete_feature:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
continuous_feature=[feature for feature in numeric_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous Feature Count {}".format(len(continuous_feature)))
for feature in continuous_feature:
    data=train.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()
categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns
train.skew(), train.kurt()
y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
sns.distplot(train.skew(),color='blue',axlabel ='Skewness')
plt.figure(figsize = (12,8))
sns.distplot(train.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)
#plt.hist(train.kurt(),orientation = 'vertical',histtype = 'bar',label ='Kurtosis', color ='blue')
plt.show()
plt.hist(train['SalePrice'],orientation = 'vertical',histtype = 'bar', color ='blue')
plt.show()
target = np.log(train['SalePrice'])
target.skew()
plt.hist(target,color='black')
correlation = numeric_features.corr()
print(correlation['SalePrice'].sort_values(ascending = False),'\n')
f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of Numeric Features with Sale Price',y=1,size=16)
sns.heatmap(correlation,square = True,  vmax=0.8)
k= 11
cols = correlation.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(train[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
sns.set()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(train[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
saleprice_overall_quality= train.pivot_table(index ='OverallQual',values = 'SalePrice', aggfunc = np.median)
saleprice_overall_quality.plot(kind = 'bar',color = 'blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.show()
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(12, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
var = 'Neighborhood'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)
var = 'SaleType'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)
var = 'SaleCondition'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)
# checking percentage of missing values
data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
features_with_na=[features for features in data.columns if data[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature, np.round(data[feature].isnull().mean(), 4),  ' % of Missing Values')
#test data
data_out = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
features_with_na=[features for features in data_out.columns if data[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature, np.round(data_out[feature].isnull().mean(), 4),  ' % of Missing Values')
# features with some missing values with sales Price
for feature in features_with_na:
    dataset = data.copy()
    dataset[feature] = np.where(dataset[feature].isnull(), 1, 0)
   
    # Calculate the mean of SalePrice where the information is missing or present
    dataset.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
train=train.drop(['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Neighborhood','LotFrontage','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrType','MasVnrArea','MSZoning','Electrical','Utilities','Functional','KitchenQual','Exterior1st','Exterior2nd','SaleType','MSSubClass'],axis=1)

train=train.dropna(axis=1)
test=test.dropna(axis=1)
train.isnull().sum()
train.info()
test.info()
#train data
sum([True for idx,row in train.iterrows() if any(row.isnull())])

#test data
sum([True for idx,row in test.iterrows() if any(row.isnull())])
#convert categorical variable into dummy
train = pd.get_dummies(train)
test = pd.get_dummies(test)
train.isnull().sum()
train.head(100)
test.head()
a = np.intersect1d(test.columns, train.columns)
print (a)
train_common=train[a]
test=test[a]
train_common.head()
X=train_common
Y=y_train
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_scale
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("X_train's shape : ",X_train.shape)
print("X_test's shape : ",X_test.shape)
print("Y_train's shape : ",Y_train.shape)
print("Y_test's shape : ",Y_test.shape)
from sklearn.linear_model import LinearRegression
model=LinearRegression(normalize=True)
model.fit(X_train,Y_train)
from sklearn.ensemble import RandomForestRegressor
rfc=RandomForestRegressor(n_estimators=10000, random_state=1, n_jobs=-1)
rfc.fit(X_train,Y_train)
# Model evaluation for training set
Y_train_pred = model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, Y_train_pred))) #root mean square error
r2 = r2_score(Y_train, Y_train_pred) # it gives the score based on the relationship between actual output and predicted output by the model

print("Model training performance:")
print("---------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")
Y_test_pred=model.predict(X_test)
# Model evaluation for testing set
B_test_pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, Y_test_pred)))
r2 = r2_score(Y_test, Y_test_pred)

print("Model testing performance:")
print("--------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
# Model evaluation for training set
Y_train_pred = rfc.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, Y_train_pred))) #root mean square error
r2 = r2_score(Y_train, Y_train_pred) # it gives the score based on the relationship between actual output and predicted output by the model



print("Model training performance:")
print("---------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")
Y_test_pred=rfc.predict(X_test)
# Model evaluation for testing set
B_test_pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, Y_test_pred)))
r2 = r2_score(Y_test, Y_test_pred)

print("Model testing performance:")
print("--------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
feat_importances = pd.Series(rfc.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
from sklearn.feature_selection import SelectFromModel
# Create a selector object that will use the random forest classifier to identify
# It will select the features based on the importance score
rf_sfm = SelectFromModel(rfc)
rf_sfm = rf_sfm.fit(X_train, Y_train)
X_important_train = rf_sfm.transform(X_train)
X_important_test = rf_sfm.transform(X_test)
X_important_train
# Create a new random forest classifier for the most important features
clf_important = RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important = clf_important.fit(X_important_train, Y_train)
# Model evaluation for training set
Y_train_pred = clf_important.predict(X_important_train)
rmse = (np.sqrt(mean_squared_error(Y_train, Y_train_pred))) #root mean square error
r2 = r2_score(Y_train, Y_train_pred) # it gives the score based on the relationship between actual output and predicted output by the model



print("Model training performance:")
print("---------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")
Y_test_pred=clf_important.predict(X_important_test)
# Model evaluation for testing set
B_test_pred = clf_important.predict(X_important_test)
rmse = (np.sqrt(mean_squared_error(Y_test, Y_test_pred)))
r2 = r2_score(Y_test, Y_test_pred)

print("Model testing performance:")
print("--------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

output_model=pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
output_model.head()
test_imp= rf_sfm.transform(test)
output=clf_important.predict(test_imp)
prediction=pd.DataFrame({'Id':test.Id,'SalePrice':output})

prediction.to_csv('prediction_c.csv',index=False)