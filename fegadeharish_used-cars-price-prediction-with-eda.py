import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df1 = pd.read_csv("../input/autos.csv", encoding="ISO-8859-1")
df = df1.copy()

print(df1.shape)

df1.head()
df1.describe().T
df1.info()
df1.isna().sum()
df1[df1['dateCrawled'].isnull()].isna().sum()

# There are 103701 rows with all NA values which we straight away remove.

df2 = df1.dropna(thresh=1)

print(df2.shape)

df2.isna().sum()
df3 = df2.dropna(thresh=12)

print(df3.shape)

df3.isna().sum()
plt.subplots(figsize=(15,10))

plt.subplot(321)

df3.seller.value_counts(100).plot(kind='bar', title="Seller Proportion", fontsize=18)

plt.xticks(rotation=0, fontsize=12)

plt.yticks(rotation=0, fontsize=12)



plt.subplot(322)

df3.offerType.value_counts(100).plot(kind='bar', title='Offer Type Proportion', fontsize=18)

plt.xticks(rotation=0, fontsize=12)

plt.yticks(rotation=0, fontsize=12)

plt.show()



plt.subplots(figsize=(15,10))

plt.subplot(323)

df3.fuelType.value_counts(100).plot(kind='bar', title='Fuel Type Proportion', fontsize=18)

plt.xticks(rotation=0, fontsize=12)

plt.yticks(rotation=0, fontsize=12)



plt.subplot(324)

df3.gearbox.value_counts(100).plot(kind='bar', title='Gearbox Proportion', fontsize=18)

plt.xticks(rotation=0, fontsize=12)

plt.yticks(rotation=0, fontsize=12)

plt.show()



plt.subplots(figsize=(15,10))

plt.subplot(325)

df3.notRepairedDamage.value_counts(100).plot(kind='bar', title='Damage Not Repaired Proportion', fontsize=18)

plt.xticks(rotation=0, fontsize=12)

plt.yticks(rotation=0, fontsize=12)



plt.subplot(326)

df3.nrOfPictures.value_counts(100).plot(kind='bar', title='Number Of Pictures Proportion', fontsize=18)

plt.xticks(rotation=0, fontsize=12)

plt.yticks(rotation=0, fontsize=12)

plt.show()
df4 = df3.drop(['seller','offerType','abtest','nrOfPictures','dateCrawled', 'dateCreated', 'lastSeen','name','postalCode'], axis=1)

df4.shape
df4.kilometer = df4.kilometer.astype('int64')
# Distribution of target variable

plt.subplots(figsize=(15,15))

plt.subplot(321)

sns.distplot(df4.price)

plt.subplot(322)

sns.boxplot(df4.price)



# Distribution of powerPS variable

plt.subplot(323)

sns.distplot(df4.powerPS)

plt.subplot(324)

sns.boxplot(df4.powerPS)



# Distribution of kilometer variable

plt.subplot(325)

sns.distplot(df4.kilometer)

plt.subplot(326)

sns.boxplot(df4.kilometer)



plt.show()
qnt = np.quantile(a=df4.price, q=[0.04,0.1,0.25,0.5,0.75,0.99,0.995,0.9999,0.99999,1])

for q in qnt:

    print('{:.2f}'.format(q))

qnt = np.quantile(a=df4.powerPS, q=[0.1,0.11,0.15,0.25,0.5,0.75,0.99,0.995,0.9999,0.99999,1])

for q in qnt:

    print('{:.2f}'.format(q))



print(df4[(df4['powerPS'] > 408) | (df4['powerPS'] <1)].shape)
qnt = np.quantile(a=df4.kilometer, q=[0.03,0.04,0.1,0.25,0.5,0.75,0.99,0.995,0.9999,0.99999,1])

for q in qnt:

    print('{:.2f}'.format(q))
print(df4[df4['price'] > 47694.75].shape)

df5 = df4[(df4['price'] <= 47695) & (df4['price'] > 50) & (df4['powerPS'] > 5) & (df4['powerPS'] < 408)]

print(df5.shape)
# Distribution of target variable

plt.subplots(figsize=(15,15))

plt.subplot(321)

sns.distplot(df5.price)

plt.subplot(322)

sns.boxplot(df5.price)



# Distribution of powerPS variable

plt.subplots(figsize=(15,15))

plt.subplot(323)

sns.distplot(df5.powerPS)

plt.subplot(324)

sns.boxplot(df5.powerPS)



# Distribution of kilometer variable

plt.subplots(figsize=(15,15))

plt.subplot(325)

sns.distplot(df5.kilometer)

plt.subplot(326)

sns.boxplot(df5.kilometer)



plt.show()
plt.subplots(figsize=(20,12))

plt.subplot(221)

sns.scatterplot(x=df5.powerPS, y=df5.price, hue=df5.vehicleType)



plt.subplot(222)

sns.scatterplot(x=df5.powerPS, y=df5.price, hue=df5.gearbox)



plt.subplot(223)

sns.scatterplot(x=df5.powerPS, y=df5.price, hue=df5.fuelType)



plt.subplot(224)

sns.scatterplot(x=df5.powerPS, y=df5.price, hue=df5.notRepairedDamage)



plt.show()
print(df5.yearOfRegistration.unique())
df6 = df5.copy()

df6.yearOfRegistration = df6.yearOfRegistration.astype('int64')

df6.kilometer = df6.kilometer.astype('int64')

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 10000)] = 2015

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 20000) & (df6['kilometer'] > 10000)] = 2014

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] > 20000)] = 2013

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 40000) & (df6['kilometer'] > 30000)] = 2012

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 50000) & (df6['kilometer'] > 40000)] = 2011

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 60000) & (df6['kilometer'] > 50000)] = 2010

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 70000) & (df6['kilometer'] > 60000)] = 2009

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 80000) & (df6['kilometer'] > 70000)] = 2008

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 90000) & (df6['kilometer'] > 80000)] = 2007

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 100000) & (df6['kilometer'] > 90000)] = 2006

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] <= 125000) & (df6['kilometer'] > 100000)] = 2005

df6.yearOfRegistration[((df6['yearOfRegistration'] > 2015) | (df6['yearOfRegistration'] < 2000)) & (df6['kilometer'] > 125000)] = 2004
plt.subplots(figsize=(20,5))

plt.subplot(121)

sns.countplot(df6.yearOfRegistration)

plt.xticks(rotation=90, fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel(s='Year Of Registration', fontsize=15)

plt.ylabel(s='Count', fontsize=15)



plt.subplot(122)

sns.scatterplot(x=df6.yearOfRegistration, y=df6.price)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel(s='Year Of Registration', fontsize=15)

plt.ylabel(s='Price', fontsize=15)

plt.show()
plt.subplots(figsize=(15,10))

plt.subplot(321)

df6.brand.value_counts(100)[:5].plot(kind='bar', title="Brand", fontsize=18)

plt.xticks(rotation=0, fontsize=12)

plt.yticks(rotation=0, fontsize=12)



plt.subplot(322)

df6.vehicleType.value_counts(100)[:5].plot(kind='bar', title='Vehicle Type', fontsize=18)

plt.xticks(rotation=0, fontsize=12)

plt.yticks(rotation=0, fontsize=12)

plt.show()



plt.subplots(figsize=(15,10))

plt.subplot(323)

df6.model.value_counts(100)[:5].plot(kind='bar', title='Model', fontsize=18)

plt.xticks(rotation=0, fontsize=12)

plt.yticks(rotation=0, fontsize=12)



plt.show()
print(df6.shape)

df6.notRepairedDamage = df6['notRepairedDamage'].fillna(method='ffill')

df6.isna().sum()
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.model_selection import train_test_split
# vehicleType column



# filter data for model building

X = df6[df6['vehicleType'].notna()].drop(['vehicleType','model'], axis=1)

y = df6.vehicleType[df6['vehicleType'].notna()]



# create test data with unknown vehicleType fields

xt = df6[df6['vehicleType'].isna()].drop(['vehicleType','model'], axis=1)



# one-hot encode the categorical variables

X_dum = pd.get_dummies(X)

xt_dum = pd.get_dummies(xt)



# build a decision tree model

dtree_veh = DecisionTreeClassifier().fit(X_dum,y)



# predict on test data

veh_pred = dtree_veh.predict(xt_dum)



# fill the missing values with the predicted values

df6.vehicleType[df6['vehicleType'].isna()] = veh_pred

df6.vehicleType.isna().sum()
# model column



# filter data for model building

X = df6[df6['model'].notna()].drop(['model'], axis=1)

y = df6.model[df6['model'].notna()]



# create test data with unknown model fields

xt = df6[df6['model'].isna()].drop(['model'], axis=1)



# one-hot encode the categorical variables

X_dum = pd.get_dummies(X)

xt_dum = pd.get_dummies(xt)

xt_dum.drop('brand_sonstige_autos', axis=1, inplace=True)



# build a decision tree model

dtree_model = DecisionTreeClassifier().fit(X_dum,y)



# predict on test data

model_pred = dtree_model.predict(xt_dum)



# fill the missing values with the predicted values

df6.model[df6['model'].isna()] = model_pred

df6.model.isna().sum()
# gearbox column



# filter data for model building

X = df6[df6['gearbox'].notna()].drop(['gearbox'], axis=1)

y = df6.gearbox[df6['gearbox'].notna()]



# create test data with unknown gearbox fields

xt = df6[df6['gearbox'].isna()].drop(['gearbox'], axis=1)



# one-hot encode the categorical variables

X_dum = pd.get_dummies(X)

xt_dum = pd.get_dummies(xt)

for col in X_dum.columns:

    if col not in xt_dum:

        xt_dum[col] = np.zeros(len(xt_dum))

        

# build a decision tree model

dtree_gear = DecisionTreeClassifier().fit(X_dum,y)



# predict on test data

gear_pred = dtree_gear.predict(xt_dum)



# fill the missing values with the predicted values

df6.gearbox[df6['gearbox'].isna()] = gear_pred

df6.gearbox.isna().sum()
# fuelType column



# filter data for model building

X = df6[df6['fuelType'].notna()].drop(['fuelType'], axis=1)

y = df6.fuelType[df6['fuelType'].notna()]



# create test data with unknown fuelType fields

xt = df6[df6['fuelType'].isna()].drop(['fuelType'], axis=1)



# one-hot encode the categorical variables

X_dum = pd.get_dummies(X)

xt_dum = pd.get_dummies(xt)

for col in X_dum.columns:

    if col not in xt_dum:

        xt_dum[col] = np.zeros(len(xt_dum))

        

# build a decision tree model

dtree_fuel = DecisionTreeClassifier().fit(X_dum,y)



# predict on test data

fuel_pred = dtree_fuel.predict(xt_dum)



# fill the missing values with the predicted values

df6.fuelType[df6['fuelType'].isna()] = fuel_pred

df6.fuelType.isna().sum()
print(df6.shape)

df6.isna().sum()
plt.subplots(figsize=(8,6))

sns.heatmap(df6.corr(), annot=True, square=True, annot_kws={'fontsize':15})

plt.show()
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler, scale

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
X = df6.drop('price', axis=1)

y = df6.price



X_dum = pd.get_dummies(X)

X_scaled = scale(X_dum)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=10)
linreg = LinearRegression().fit(X_train, y_train)

y_pred = linreg.predict(X_test)

print("R2_Score for linear regression: {:.2f}".format(linreg.score(X_test, y_test)))

print("RMSE score: {}".format(np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))))



mse_all = cross_val_score(linreg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)

reg_cv = cross_val_score(linreg, X_train, y_train, cv=5)

mean_mse_linreg = np.mean(mse_all)

print("mean MSE with 5k cross validation linear regression: {:.2f}".format(mean_mse_linreg))

print("R2_score with 5k cross validation linear regression: {:.2f}".format(np.mean(reg_cv)))
ridge = Ridge(alpha=0.05, normalize=True)



ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

print("R2_score with Ridge regression: {:.2f}".format(ridge.score(X_test, y_test)))



# Perform 5-fold cross-validation: ridge_cv

ridge_cv = cross_val_score(ridge, X_train, y_train, cv=5)

# Print the cross-validated scores

print("R2_score with 5k cross validation Ridge regression: {:.2f}".format(np.mean(ridge_cv)))

lasso = Lasso(alpha=0.4, normalize=True)



# Fit the regressor to the data

lasso.fit(X_train,y_train)



# Compute and print the coefficients

lasso_coef = lasso.coef_.tolist()

print(lasso_coef)
df6.skew()
from scipy.stats import norm

sns.distplot(df6.price, fit=norm)

print(df6.price.skew())

print(norm.fit(df6.price))
from scipy import stats



stats.probplot(df6.price, plot=plt)
df7 = df6.copy()

df7.price = np.log(df6.price)
sns.distplot(df7.price, fit=norm)

print(df7.price.skew())

print(norm.fit(df7.price))

plt.show()



stats.probplot(df7.price, plot=plt)



plt.show()
# df7 = 



X_dum = pd.get_dummies(df7)



X = X_dum.drop('price', axis=1)

y = X_dum.price



# X_scaled = scale(X_dum)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
ridge = Ridge(alpha=0.05, normalize=True)



ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

print("R2_score with Ridge regression: {:.2f}".format(ridge.score(X_test, y_test)))

# print("Ridge best parameter: {:.2f}".format(ridge.best_params_))

# print("Ridge best R2_score: {:.2f}".format(ridge.best_score_))



# Perform 5-fold cross-validation: ridge_cv

ridge_cv = cross_val_score(ridge, X_train, y_train, cv=5)

# Print the cross-validated scores

print("R2_score with 5k cross validation Ridge regression: {:.2f}".format(np.mean(ridge_cv)))

ridge.score(X_train,y_train)
lasso = Lasso(alpha=0.4, normalize=True)



# Fit the regressor to the data

lasso.fit(X_train,y_train)



# Compute and print the coefficients

lasso_coef = lasso.coef_.tolist()

print(lasso_coef)

y_lasspred = lasso.predict(X_test)

lasso.score(X_test,y_test)