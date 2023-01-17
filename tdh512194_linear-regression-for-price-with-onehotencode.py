import pandas as pd
import numpy as np
data = pd.read_csv('../input/Melbourne_housing_FULL.csv')
pd.set_option('display.max_columns', None) # display all columns
data.head(5)
data.describe(include='all').T
data.shape
data.drop(columns=['Lattitude','Longtitude'], inplace=True)
data.Distance.value_counts(dropna=False)
data.Distance.fillna(data.Distance.mode(), inplace=True)
data[data.Postcode.isnull()]
data.drop(index=29483, inplace=True)
data.reset_index().drop(columns='index', inplace=True)
data.iloc[29483]
data.Regionname.value_counts(dropna=False)
data[data.Regionname.isnull()]
data.drop(index=[18523, 26888], inplace=True)
data.reset_index().drop(columns='index', inplace=True)
data.describe(include='all').T
data.YearBuilt[data.YearBuilt >= 2020].value_counts(dropna=False)
data.YearBuilt.replace(2106, 2016, inplace=True)
data.YearBuilt.value_counts(dropna=False)
data.YearBuilt.corr(data.Price)
data.drop(columns='YearBuilt', inplace=True)
data.BuildingArea.value_counts(dropna=False)
data = data[pd.notnull(data['BuildingArea'])]
data.loc[data.BuildingArea.idxmax()]
data.drop(index=data.BuildingArea.idxmax(), inplace=True)
data.describe(include='all').T
data.Landsize.value_counts(dropna=False)
data.Landsize.fillna(value=0, inplace=True) # using mode() does not work -> resort to hardcode value 0
data.Car.value_counts(dropna=False)
data.Car.fillna(value=0, inplace=True)
import copy
data_regr = copy.copy(data[pd.notnull(data['Price'])])
data_regr.describe(include='all').T
data.Rooms.value_counts()
data.Type.value_counts()
data.Method.value_counts()
data.CouncilArea.value_counts()
data.describe().T
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(theme='grade3')
corr = data[['Rooms', 'Price', 'Distance', 
             'Postcode', 'Bedroom2', 'Bathroom', 
             'Car', 'Landsize', 'BuildingArea', 'Propertycount']].corr()
sns.heatmap(corr)
data[['Rooms','Bedroom2','Bathroom','Price']].corr()
data.drop(columns='Bedroom2', inplace=True)
data_regr.drop(columns='Bedroom2', inplace=True)
data.describe(include='all').T
#TODO: Date, methode, type
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error
data_regr.describe(include=['O']).T
data_regr.drop(columns=['Suburb','Postcode'], inplace=True)
import datetime
def to_year(date_str):
    return datetime.datetime.strptime(date_str.strip(),'%d/%m/%Y').year
data_regr['Date'] = data_regr.Date.apply(to_year)
data_regr.Date.value_counts()
import re
def to_street(str):
    return re.sub('[^A-Za-z]+', '', str)
data_regr.Address.apply(to_street).value_counts().count()
data_regr.drop(columns='Address', inplace=True)
counts = data_regr.SellerG.value_counts()
counts
data_regr.SellerG[data['SellerG'].isin(counts[counts < 100].index)] = 'less than 100'
data_regr.SellerG[data['SellerG'].isin(counts[(counts >= 100) & (counts < 200)].index)] = '100 - 200'
data_regr.SellerG[data['SellerG'].isin(counts[(counts >= 200) & (counts < 500)].index)] = '200 - 500'
data_regr.SellerG[data['SellerG'].isin(counts[(counts >= 500) & (counts < 1000)].index)] = '500 - 1000'
data_regr.SellerG[data['SellerG'].isin(counts[counts > 1000].index)] = 'over 1000'
data_regr.SellerG.value_counts()
data_regr.drop(columns='CouncilArea', inplace=True)
data_regr.describe(include=['O']).T
data.head()
data = data.reset_index().drop(columns='index') # do not use inplace=True if combine
data_regr = data_regr.reset_index().drop(columns='index')
data_regr.head()
categoricals = ['Type', 'Method', 'SellerG', 'Regionname', 'Date']
for feature in categoricals:
    df = copy.copy(pd.get_dummies(data_regr[feature], drop_first=True))
    data_regr = pd.concat([data_regr, df], axis=1)
    data_regr.drop(columns=feature, inplace=True)
data_regr.head()
data_regr.shape
model_HO = linear_model.LinearRegression()
train, test = train_test_split(data_regr, test_size = 0.2, random_state=512)
train.shape
test.shape
X_train = train.loc[:, data_regr.columns != 'Price']
y_train = train.Price

X_test = test.loc[:, data_regr.columns != 'Price']
y_test = test.Price
model_HO.fit(X_train.values, y_train.values)
predict_train = model_HO.predict(X_train.values)
mean_squared_error(y_train, predict_train)
predict_test = model_HO.predict(X_test.values)
mean_squared_error(y_test, predict_test)
fig, ax = plt.subplots()
ax.scatter(y_test, predict_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
model_CV = linear_model.LinearRegression()
y = data_regr.Price
X = data_regr.loc[:, data_regr.columns != 'Price']
predicted = cross_val_predict(model_CV, X.values, y.values, cv=5)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
mean_squared_error(y.values, predicted)
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_new = pd.DataFrame(pca.fit_transform(X.values))
model_HO_PCA = linear_model.LinearRegression()
dataPCA = pd.concat([X_new, y], axis=1)
train, test = train_test_split(dataPCA, test_size = 0.2, random_state=512)
X_train = train.loc[:, train.columns != 'Price']
y_train = train.Price

X_test = test.loc[:, test.columns != 'Price']
y_test = test.Price
model_HO_PCA.fit(X_train.values, y_train.values)
predict_train = model_HO_PCA.predict(X_train.values)
mean_squared_error(y_train, predict_train)
predict_test = model_HO_PCA.predict(X_test.values)
mean_squared_error(y_test, predict_test)
fig, ax = plt.subplots()
ax.scatter(y_test, predict_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
model_CV_PCA = linear_model.LinearRegression()
y = dataPCA.Price
X = dataPCA.loc[:, dataPCA.columns != 'Price']
predicted = cross_val_predict(model_CV_PCA, X.values, y.values, cv=5)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
mean_squared_error(y.values, predicted)
X = data_regr.loc[:, data_regr.columns != 'Price']
y = data_regr.Price
model = linear_model.LinearRegression()
model.fit(X.values, y.values)
predict = model.predict(X.values)
sns.residplot(predict, y.values)
a = (y.values - predict)
fig, ax = plt.subplots()
ax.scatter(data_regr.Rooms.values, a)
ax.set_xlabel('Rooms')
ax.set_ylabel('Residual')
plt.show()
fig, ax = plt.subplots()
ax.scatter(data_regr.Distance.values, a)
ax.set_xlabel('Distance')
ax.set_ylabel('Residual')
plt.show()
fig, ax = plt.subplots()
ax.scatter(data_regr.Bathroom.values, a)
ax.set_xlabel('Bathroom')
ax.set_ylabel('Residual')
plt.show()
fig, ax = plt.subplots()
ax.scatter(data_regr.Car.values, a)
ax.set_xlabel('Car')
ax.set_ylabel('Residual')
plt.show()
fig, ax = plt.subplots()
ax.scatter(data_regr.Landsize.values, a)
ax.set_xlabel('Landsize')
ax.set_ylabel('Residual')
plt.show()
fig, ax = plt.subplots()
ax.scatter(data_regr.BuildingArea.values, a)
ax.set_xlabel('BuildingArea')
ax.set_ylabel('Residual')
plt.show()
fig, ax = plt.subplots()
ax.scatter(data_regr.Propertycount.values, a)
ax.set_xlabel('Propertycount')
ax.set_ylabel('Residual')
plt.show()
data_regr = data_regr[data_regr.BuildingArea < 3000]
data_regr = data_regr.reset_index()
data_regr.drop(columns='index', inplace=True)
data_regr.describe().T
data_regr = data_regr[data_regr.Landsize < 3000]
data_regr = data_regr.reset_index()
data_regr.drop(columns='index', inplace=True)
X = data_regr.loc[:, data_regr.columns != 'Price']
y = data_regr.Price
model = linear_model.LinearRegression()
model.fit(X.values, y.values)
predict = model.predict(X.values)
mean_squared_error(y.values, predict)
fig, ax = plt.subplots()
ax.scatter(y, predict)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
