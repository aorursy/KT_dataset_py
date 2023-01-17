import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from scipy.stats import norm
from scipy import stats



df = pd.read_csv('../input/train.csv')
df.shape
df.head()
df.columns
df.dtypes
df['SalePrice'].describe()
import matplotlib.pyplot as plt

plt.hist(df['SalePrice'], histtype='bar',  rwidth=0.8)
plt.xlabel("House SalePrice")
plt.show()
x = df['GarageArea']
y = df['SalePrice']
plt.scatter(x, y, s=75, c=y, alpha=0.75)
plt.xlabel('GarageArea ')
plt.ylabel('House_SalePrice')
plt.show()
#sales price vs Lot Area
x = df['PoolArea']
y = df['SalePrice']
plt.scatter(x, y, s=75, c=y, alpha=0.5)
plt.xlabel('PoolArea')
plt.ylabel('House_SalePrice')

plt.show()
#sales price vs TotalBsmtSF
x = df['TotalBsmtSF']
y = df['SalePrice']
plt.scatter(x, y, s=75, c=y, alpha=1)
plt.xlabel('TotalBsmtSF')
plt.ylabel('House_SalePrice')

plt.show()
#House_price vs Over_all_Quality
data = pd.concat([df['SalePrice'], df['OverallQual']], axis=1)


f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()
#House Price vs YearBuilt
var = 'YearBuilt'

data = pd.concat([df['SalePrice'], df['YearBuilt']], axis=1)

f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()
#Corelation Matrix
corrmat = df.corr()
#print(corrmat)
f, ax = plt.subplots(figsize=(12, 9))
#control the positive and negative correlation with vmin, vmax
sns.heatmap(corrmat, vmin=0, vmax =1, square=True)
plt.show()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, 
                 fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', \
        'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[cols], size=2.5)
plt.show()

total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_clean = df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', \
         'LotFrontage', 'GarageCond', 'GarageType', 'GarageType', \
        'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual',\
        'MasVnrArea', 'MasVnrType', 'Electrical', 'GarageYrBlt', 'GarageFinish', \
                  'GarageQual' ], 1)
total_missing = df_clean.isnull().sum().sort_values(ascending=False)
print(total_missing)
import seaborn as sns

sns.distplot(df_clean['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_clean['SalePrice'], plot=plt)
plt.show()
#applying log transformation
df_clean['SalePrice'] = np.log(df_clean['SalePrice'])
sns.distplot(df_clean['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_clean['SalePrice'], plot=plt)
plt.show()
df_clean_final = pd.get_dummies(df_clean)
df_clean_final.head()
#linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Dropping the Label Variable and Keeping only Features
#SalePrice - Dependent or Label Variable

X = df_clean_final.drop(['SalePrice'], 1)
y = df_clean_final['SalePrice']

#Splitting the Features and Labels for Training and Testing
#We are Splitting the Training size and Testing size with 80:20
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Calling the LinearRegression Function
regr = LinearRegression()

#.fit train the model
clf = regr.fit(x_train, y_train)

#y=mx+c = co-efficients for each and every features
clf_score = clf.score(x_test, y_test)
print(clf_score)
clf_predict = clf.predict(x_test)
print(clf_predict[:5])
print(y_test[:5])
#ploting a reg graph with sns
rg = sns.regplot(clf_predict, y_test, color='blue')
plt.show()
clf_predict_price = pd.DataFrame(clf_predict)
x_test_result = pd.DataFrame(x_test)
x_test_result.head()
x_test_result["true_sales_price"] = y_test
x_test_result["predicted_sales_price"] = clf_predict_price

x_test_result.head()
x_test_result.to_csv("Housing_price_prediction_liner_reg.csv")

from sklearn.ensemble import GradientBoostingRegressor


#Dropping the Label Variable and Keeping only Features
#SalePrice - Dependent or Label Variable

X = df_clean_final.drop(['SalePrice'], 1)
y = df_clean_final['SalePrice']

#Splitting the Features and Labels for Training and Testing
#We are Splitting the Training size and Testing size with 80:20
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



model = GradientBoostingRegressor()
clf = model.fit(x_train, y_train)
clf_score = clf.score(x_test, y_test)
print(clf_score)

feature = df_clean['OverallQual']

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(feature, y, test_size=0.2)

#Reshape the Variable Incase of only 1 feature
x_train_s = x_train_s.values.reshape(-1, 1)
x_test_s = x_test_s.values.reshape(-1, 1)


clf = model.fit(x_train_s, y_train_s)
clf_score = clf.score(x_test_s, y_test_s)
print(clf_score)
feature = df_clean['GrLivArea']

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(feature, y, test_size=0.2)

x_train_s = x_train_s.values.reshape(-1, 1)
x_test_s = x_test_s.values.reshape(-1, 1)


clf = model.fit(x_train_s, y_train_s)
clf_score = clf.score(x_test_s, y_test_s)
print(clf_score)
feature = df_clean['TotalBsmtSF']

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(feature, y, test_size=0.2)

x_train_s = x_train_s.values.reshape(-1, 1)
x_test_s = x_test_s.values.reshape(-1, 1)


clf = model.fit(x_train_s, y_train_s)
clf_score = clf.score(x_test_s, y_test_s)
print(clf_score)
feature = df_clean['GarageCars']

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(feature, y, test_size=0.2)

x_train_s = x_train_s.values.reshape(-1, 1)
x_test_s = x_test_s.values.reshape(-1, 1)


clf = model.fit(x_train_s, y_train_s)
clf_score = clf.score(x_test_s, y_test_s)
print(clf_score)
feature = df_clean['GarageArea']

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(feature, y, test_size=0.2)

x_train_s = x_train_s.values.reshape(-1, 1)
x_test_s = x_test_s.values.reshape(-1, 1)


clf = model.fit(x_train_s, y_train_s)
clf_score = clf.score(x_test_s, y_test_s)
print(clf_score)
feature = df_clean['TotRmsAbvGrd']

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(feature, y, test_size=0.2)

x_train_s = x_train_s.values.reshape(-1, 1)
x_test_s = x_test_s.values.reshape(-1, 1)


clf = model.fit(x_train_s, y_train_s)
clf_score = clf.score(x_test_s, y_test_s)
print(clf_score)
feature = df_clean['1stFlrSF']

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(feature, y, test_size=0.2)

x_train_s = x_train_s.values.reshape(-1, 1)
x_test_s = x_test_s.values.reshape(-1, 1)


clf = model.fit(x_train_s, y_train_s)
clf_score = clf.score(x_test_s, y_test_s)
print(clf_score)
feature = df_clean['YearBuilt']

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(feature, y, test_size=0.2)

x_train_s = x_train_s.values.reshape(-1, 1)
x_test_s = x_test_s.values.reshape(-1, 1)


clf = model.fit(x_train_s, y_train_s)
clf_score = clf.score(x_test_s, y_test_s)
print(clf_score)

from sklearn.model_selection import train_test_split

#Total BasementSF is highly Correlated with 1stFlrSF, so onbehalf driping
#both the Variables we are dropping 1stFlrSF due to relss predictive power 
#over the Sales Price

X = df_clean_final.drop(['SalePrice', \
                        'TotRmsAbvGrd', \
                        ], 1)
y = df_clean_final['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
clf = model.fit(x_train, y_train)
clf_score = clf.score(x_test, y_test)
print(clf_score)
from sklearn.preprocessing import RobustScaler

#Transforming the variables with RobustScaling
out_robust = RobustScaler()
X = out_robust.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
clf = model.fit(x_train, y_train)
clf_score = clf.score(x_test, y_test)
print(clf_score)
#Model Prediction
clf_predict_nl_reg = clf.predict(x_test)
print(clf_predict_nl_reg[:5])
print(y_test[:5])
#ploting a reg graph with sns
rg = sns.regplot(clf_predict_nl_reg, y_test, color='blue')
plt.show()
