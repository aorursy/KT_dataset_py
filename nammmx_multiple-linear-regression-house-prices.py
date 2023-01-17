import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.isna().sum()
df.dtypes
df.drop(columns=['id', 'date'], inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn import preprocessing
X = df.drop(columns=['price'])

y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
model_lr = LinearRegression()

model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)

print("Training set score: {:.7f}".format(model_lr.score(X_train, y_train)))

print("Test set score: {:.7f}".format(model_lr.score(X_test, y_test)))

print("RMSE: {:.7f}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
def calculate_residuals(model, features, label):

    predictions = model_lr.predict(features)

    df_result = pd.DataFrame({'Actual':label, 'Predicted':predictions})

    df_result['Residuals'] = abs(df_result['Actual']) - abs(df_result['Predicted'])

    return df_result
def linear_assumption(model, features, label):

    df_result = calculate_residuals(model, features, label)

    fig1, ax1 = plt.subplots(figsize=(12,8))

    ax1 = sns.regplot(x='Actual', y='Predicted', data=df_result, color='steelblue')

    line_coords = np.arange(df_result.min().min(), df_result.max().max())

    ax1 = plt.plot(line_coords, line_coords,  # X and y points

              color='indianred')
linear_assumption(model_lr, X_test, y_test)
df_result = calculate_residuals(model_lr, X_test, y_test)

fig2, ax2 = plt.subplots(figsize=(12,8))

ax2.scatter(x=df_result['Predicted'], y=df_result['Residuals'], color='steelblue')

plt.axhline(y=0, color='indianred')

ax2.set_ylabel('Residuals', fontsize=12)

ax2.set_xlabel('Predicted', fontsize=12)

plt.show()
plt.style.use('ggplot')

fig3, ax3 = plt.subplots(figsize=(15,4))

ax3 = sns.boxplot(x=df['price'], color='steelblue')
df1 = df[~(df['price']>4000000)]

df1
X1 = df1.drop(columns=['price'])

y1 = df1['price']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=10)
model_lr1 = LinearRegression()

model_lr1.fit(X1_train, y1_train)

y1_pred = model_lr1.predict(X1_test)

print("Training set score: {:.7f}".format(model_lr1.score(X1_train, y1_train)))

print("Test set score: {:.7f}".format(model_lr1.score(X1_test, y1_test)))

print("RMSE: {:.7f}".format(np.sqrt(metrics.mean_squared_error(y1_test, y1_pred))))
linear_assumption(model_lr1, X1_test, y1_test)
df_result = calculate_residuals(model_lr1, X1_test, y1_test)

fig4, ax4 = plt.subplots(figsize=(12,8))

ax4.scatter(x=df_result['Predicted'], y=df_result['Residuals'], color='steelblue')

ax4.set_ylabel('Residuals', fontsize=12)

ax4.set_xlabel('Predicted', fontsize=12)

plt.axhline(y=0, color='indianred')

plt.show()
plt.style.use('ggplot')

sns.pairplot(df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'view', 'grade']],  

             y_vars=['price'], x_vars=['bedrooms', 'bathrooms', 'sqft_living', 'view', 'grade'], 

             height=5, plot_kws={'color':'steelblue'}) 

plt.show()
plt.style.use('ggplot')

sns.pairplot(df[['price','sqft_above', 'sqft_basement', 'lat', 'sqft_living15']],  

             y_vars=['price'], x_vars=['sqft_above', 'sqft_basement', 'lat', 'sqft_living15'], height=5,

             plot_kws={'color':'steelblue'}) 

plt.show()
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)

X_train_poly = poly.fit_transform(X1_train)

X_test_poly = poly.fit_transform(X1_test)
model_lr_poly = LinearRegression()

model_lr_poly.fit(X_train_poly, y1_train)

y_pred_poly = model_lr_poly.predict(X_test_poly)

print("Training set score: {:.7f}".format(model_lr_poly.score(X_train_poly, y1_train)))

print("Test set score: {:.7f}".format(model_lr_poly.score(X_test_poly, y1_test)))

print(np.sqrt(metrics.mean_squared_error(y1_test, y_pred_poly)))
fig10, ax10 = plt.subplots(figsize=(12,8))

ax10 = sns.regplot(x=y1_test, y=y_pred_poly, color='steelblue')

line_coords = np.arange(df_result.min().min(), df_result.max().max())

plt.plot(line_coords, line_coords, color='indianred')

ax10.set_ylabel('Predicted', fontsize=12)

ax10.set_xlabel('Actual', fontsize=12)

plt.show()
df_result = calculate_residuals(model_lr1, X1_test, y1_test)

fig11, ax11 = plt.subplots(figsize=(12,8))

ax11.scatter(x=y_pred_poly, y=y1_test-y_pred_poly, color='steelblue')

ax11.set_ylabel('Residuals', fontsize=12)

ax11.set_xlabel('Predicted', fontsize=12)

plt.axhline(y=0, color='indianred')

plt.show()
X = df1.drop(columns=['price'])

price_trans = np.log1p(df1['price'])

X2_train, X2_test, y2_train, y2_test = train_test_split(X, price_trans, test_size=0.3, random_state=10)
poly = PolynomialFeatures(2)

X2_train_poly = poly.fit_transform(X2_train)

X2_test_poly = poly.fit_transform(X2_test)
model_lr_poly2 = LinearRegression()

model_lr_poly2.fit(X2_train_poly, y2_train)

y2_pred_poly = model_lr_poly2.predict(X2_test_poly)

print("Training set score: {:.7f}".format(model_lr_poly2.score(X2_train_poly, y2_train)))

print("Test set score: {:.7f}".format(model_lr_poly2.score(X2_test_poly, y2_test)))
fig13, ax13 = plt.subplots(figsize=(12,8))

ax13.scatter(x=y2_pred_poly, y=y2_test-y2_pred_poly, color='steelblue')

ax13.set_ylabel('Residuals', fontsize=12)

ax13.set_xlabel('Predicted', fontsize=12)

plt.axhline(y=0, color='indianred')

plt.show()
fig15, ax15 = plt.subplots(figsize=(12,8))

sns.distplot(y2_test-y2_pred_poly, color='steelblue')

plt.show()
from scipy import stats

fig16, ax16 = plt.subplots(figsize=(8,5))

stats.probplot(y2_test-y2_pred_poly, plot=plt)

plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

# from statsmodels.tools.tools import add_constant



vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(X_test.values, i) for i in range(X_test.shape[1])]

vif["features"] = X_test.columns
vif['VIF'] = vif['VIF'].apply(lambda x: "{:.2f}".format(x))

vif