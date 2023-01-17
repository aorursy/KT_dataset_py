import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Load data

data = pd.read_csv(r'/kaggle/input/abalone-dataset/abalone.csv')
print(data.shape)

print('----------')

data.info()
sns.pairplot(data)

plt.show()
print(data.isnull().sum())

print('------------------')

print(data.eq(0).sum())

data.Height = data.Height.replace(0, np.mean(data.Height))

data.Height.eq(0).sum()
fig, axes = plt.subplots(figsize = (10,10))

sns.lineplot(x = data.Rings, y= data.Length, hue = data.Sex, ax = axes)

plt.show()
fig,axes = plt.subplots(1,1, figsize = (15,8))

sns.boxplot(data = data, ax = axes, orient = 'h')

plt.show()
for i in data.select_dtypes(exclude = 'object'):

    iqr = data[i].quantile(0.75) - data[i].quantile(0.25)

    up = data[i].quantile(0.75) + (1.5 * iqr)

    low = data[i].quantile(0.25) - (1.5 * iqr)

    for j in range(data.shape[0]):

        if data[i][j] < low:

            data.replace(data[i][j], low, inplace = True)

        elif data[i][j] > up:

            data.replace(data[i][j], up, inplace = True)
fig,axes = plt.subplots(1,1, figsize = (15,8))

sns.boxplot(data = data, ax = axes, orient = 'h')

plt.show()          
sns.distplot(data.Rings)

plt.show()
data.head()
data = pd.get_dummies(data, drop_first= True)

Y = data['Rings']

X = data.drop('Rings', axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25, random_state= 0)
from sklearn.metrics import mean_squared_error, r2_score

def model(obj, X1, Y1, X2, Y2):

    obj.fit(X1, Y1)

    Y_pred = obj.predict(X2)

    mse = mean_squared_error(Y2, Y_pred)

    r2 = r2_score(Y2, Y_pred)

    return(Y_pred, mse, r2)

from sklearn.linear_model import LinearRegression

lng = LinearRegression()

Y_pred1, mse1, r2_1 = model(lng, X_train, Y_train, X_test, Y_test)

residue = Y_test - Y_pred1

print("MSE = ",mse1, " R2 score = ", r2_1)

sns.regplot(residue, Y_pred1, line_kws= {'color': 'red'} ,lowess=True)

plt.show()
Y_train = Y_train**(1/9)

Y_test = Y_test**(1/9)

Y_pred2, mse2, r2_2 = model(lng, X_train, Y_train, X_test, Y_test)

residue = Y_test - Y_pred2

print("MSE = ",mse2, " R2 score = ", r2_2)

sns.regplot(residue, Y_pred2, line_kws= {'color': 'red'} ,lowess=True)

plt.show()
fig, axes = plt.subplots(1,2)

from statsmodels.graphics.gofplots import qqplot

qqplot(residue, line = 's', ax = axes[0])

sns.distplot(residue, ax = axes[1])

plt.show()

fig, axes = plt.subplots(1,1, figsize=(10,10))

z = data.corr()

sns.heatmap(z, mask= (z>-0.3) & (z<0.3), annot = True, ax = axes)

plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

X1 = X.drop(['Sex_I', 'Sex_M'], axis= 1)

X1['Intercept'] = 1

# Compute and view VIF

vif = pd.DataFrame()

vif["variables"] = X1.columns

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X1.shape[1])]



# View results using print

print(vif)

# KNN

from sklearn import neighbors

rmse_val = []

for K in range(1,30):

    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, Y_train)  

    pred=model.predict(X_test)

    error = (mean_squared_error(Y_test,pred)) 

    rmse_val.append(error) 

curve = pd.DataFrame(rmse_val)  

curve.plot()

knn = neighbors.KNeighborsRegressor(n_neighbors = 5)

knn.fit(X_train, Y_train)  #fit the model

Y_pred4 = knn.predict(X_test) #make prediction on test set

mse4 = mean_squared_error(Y_test, Y_pred4)

r2_4 = r2_score(Y_test, Y_pred4)

print('MSE = ', mse4, 'R2 score = ', r2_4)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(interaction_only = True, degree = 2)

Xp_train = poly.fit_transform(X_train)

Xp_test = poly.transform(X_test)

knn.fit(Xp_train, Y_train)  #fit the model

Y_pred5 = knn.predict(Xp_test) #make prediction on test set

mse5 = mean_squared_error(Y_test, Y_pred5)

r2_5 = r2_score(Y_test, Y_pred5)

print('MSE = ', mse5, 'R2 score = ', r2_5)