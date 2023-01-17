#importing numpy and pandas, seaborn



import numpy as np #linear algebra

import pandas as pd #datapreprocessing, CSV file I/O

import seaborn as sns #for plotting graphs

import matplotlib.pyplot as plt
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df.info()
df.head()
#finding no of rows and columns



df.shape
df.isnull().sum()
df['bedrooms'].value_counts()
df['waterfront'].value_counts()
df['grade'].value_counts()
df['condition'].value_counts()
sns.countplot(df.bedrooms,order=df['bedrooms'].value_counts().index)
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(15,10))

plt.title('house prices by sqft_living')

plt.xlabel('sqft_living')

plt.ylabel('house prices')

plt.legend()

sns.barplot(x='sqft_living',y='price',data=df)

fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(15,10))

plt.title("house prices by sqft_above")

plt.xlabel('sqft_above')

plt.ylabel('house prices')

plt.legend()

sns.barplot(x='sqft_above',y='price',data=df)
plt.hist('sqft_living',data=df,bins=5)
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(15,10))

sns.distplot(df['sqft_living'],hist=True,kde=True,rug=False,label='sqft_living',norm_hist=True)
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(15,10))

sns.distplot(df['sqft_above'],hist=True,kde=True,rug=False,label='sqft_above',norm_hist=True)
print('Mean',round(df['sqft_living'].mean(),2))

print('Median',df['sqft_living'].median())

print('Mode',df['sqft_living'].mode()[0])
len(df[df['sqft_living']==1300])
def correlation_heatmap(df1):

    _,ax=plt.subplots(figsize=(15,10))

    colormap=sns.diverging_palette(220,10,as_cmap=True)

    sns.heatmap(df.corr(),annot=True,cmap=colormap)

    

correlation_heatmap(df)
from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline

train_data,test_data=train_test_split(df,train_size=0.8,random_state=3)

reg=linear_model.LinearRegression()

x_train=np.array(train_data['sqft_living']).reshape(-1,1)

y_train=np.array(train_data['price']).reshape(-1,1)

reg.fit(x_train,y_train)



x_test=np.array(test_data['sqft_living']).reshape(-1,1)

y_test=np.array(test_data['price']).reshape(-1,1)

pred=reg.predict(x_test)

print('linear model')

mean_squared_error=metrics.mean_squared_error(y_test,pred)

print('Sqaured mean error', round(np.sqrt(mean_squared_error),2))

print('R squared training',round(reg.score(x_train,y_train),3))

print('R sqaured testing',round(reg.score(x_test,y_test),3) )

print('intercept',reg.intercept_)

print('coefficient',reg.coef_)
_, ax = plt.subplots(figsize= (12, 10))

plt.scatter(x_test, y_test, color= 'darkgreen', label = 'data')

plt.plot(x_test, reg.predict(x_test), color='red', label= ' Predicted Regression line')

plt.xlabel('Living Space (sqft)')

plt.ylabel('price')

plt.legend()

plt.gca().spines['right'].set_visible(False)

plt.gca().spines['right'].set_visible(False)
train_data,test_data=train_test_split(df,train_size=0.8,random_state=3)

reg=linear_model.LinearRegression()

x_train=np.array(train_data['grade']).reshape(-1,1)

y_train=np.array(train_data['price']).reshape(-1,1)

reg.fit(x_train,y_train)



x_test=np.array(test_data['grade']).reshape(-1,1)

y_test=np.array(test_data['price']).reshape(-1,1)

pred=reg.predict(x_test)

print('linear model')

mean_squared_error=metrics.mean_squared_error(y_test,pred)

print('squared mean error',round(np.sqrt(mean_squared_error),2))

print('R squared training',round(reg.score(x_train,y_train),3))

print('R squared testing',round(reg.score(x_test,y_test),3))

print('intercept',reg.intercept_)

print('coeeficient',reg.coef_)
fig,ax=plt.subplots(2,1,figsize=(15,10))

sns.boxplot(x=train_data['grade'],y=train_data['price'],ax=ax[0])

sns.boxplot(x=train_data['bedrooms'],y=train_data['price'],ax=ax[1])

_ , axes = plt.subplots(1, 1, figsize=(15,10))

sns.boxplot(x=train_data['bathrooms'],y=train_data['price'])
features1=['bedrooms','grade','sqft_living','sqft_above']

reg=linear_model.LinearRegression()

reg.fit(train_data[features1],train_data['price'])

pred=reg.predict(test_data[features1])

print('complex_model 1')

mean_squared_error=metrics.mean_squared_error(y_test,pred)

print('mean squared error(MSE)', round(np.sqrt(mean_squared_error),2))

print('R squared training',round(reg.score(train_data[features1],train_data['price']),3))

print('R squared training', round(reg.score(test_data[features1],test_data['price']),3))

print('Intercept: ', reg.intercept_)

print('Coefficient:', reg.coef_)
features1 = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','grade','sqft_above','sqft_basement','lat','sqft_living15']

reg= linear_model.LinearRegression()

reg.fit(train_data[features1],train_data['price'])

pred = reg.predict(test_data[features1])

print('Complex Model_2')

mean_squared_error = metrics.mean_squared_error(y_test, pred)

print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))

print('R-squared (training) ', round(reg.score(train_data[features1], train_data['price']), 3))

print('R-squared (testing) ', round(reg.score(test_data[features1], test_data['price']), 3))

print('Intercept: ', reg.intercept_)

print('Coefficient:', reg.coef_)
polyfeat=PolynomialFeatures(degree=2)

xtrain_poly=polyfeat.fit_transform(train_data[features1])

xtest_poly=polyfeat.fit_transform(test_data[features1])



poly=linear_model.LinearRegression()

poly.fit(xtrain_poly,train_data['price'])

polypred=poly.predict(xtest_poly)



print('Complex Model_3')

mean_squared_error = metrics.mean_squared_error(test_data['price'], polypred)

print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))

print('R-squared (training) ', round(poly.score(xtrain_poly, train_data['price']), 3))

print('R-squared (testing) ', round(poly.score(xtest_poly, test_data['price']), 3))
polyfeat=PolynomialFeatures(degree=3)

xtrain_poly=polyfeat.fit_transform(train_data[features1])

xtest_poly=polyfeat.fit_transform(test_data[features1])



poly=linear_model.LinearRegression()

poly.fit(xtrain_poly,train_data['price'])

polypred=poly.predict(xtest_poly)



print('complex model_4')

mean_squared_error=metrics.mean_squared_error(test_data['price'],polypred)

print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))

print('R-squared (training) ', round(poly.score(xtrain_poly, train_data['price']), 3))

print('R-squared (testing) ', round(poly.score(xtest_poly, test_data['price']), 3))