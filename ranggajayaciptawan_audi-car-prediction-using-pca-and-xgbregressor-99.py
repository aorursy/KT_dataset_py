import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.signal import savgol_filter

from statsmodels.sandbox.stats.runs import runstest_1samp

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv')

data
data_encode_dummy = pd.get_dummies(data,columns=['model', 'transmission','fuelType'], drop_first = True)

data_encode_dummy
unique_model = data['model'].unique()

transmission_unique = data['transmission'].unique()

fueltype_unique = data['fuelType'].unique()



unique_model.sort()

transmission_unique.sort()

fueltype_unique.sort()



print(data_encode_dummy.columns)

print(unique_model)

print(transmission_unique)

print(fueltype_unique)
X = data_encode_dummy.drop(columns = ['price'])

Y = data_encode_dummy['price']



title = ['Price vs Year', 

         'Price vs Mileage', 

         'Price vs Tax', 

         'Price vs Mpg', 

         'Price vs Engine Size']



fig,ax = plt.subplots(3,2,figsize=(20,20))



i = 0

for rows in range(3):

    for cols in range(2):

        if rows == 2 and cols == 1:

            fig.delaxes(ax[rows,cols])

            break

        ax[rows,cols].scatter(x = X[X.columns[i]], y = Y)

        ax[rows,cols].set_title(title[i])

        i = i+1

        

fig.subplots_adjust(hspace=0.5, wspace=0.5)
data_encode_dummy_transform = data_encode_dummy.copy()

data_encode_dummy_transform['price'] = np.log(data_encode_dummy_transform['price'])

data_encode_dummy_transform
X = data_encode_dummy_transform.drop(columns = ['price'])

Y = data_encode_dummy_transform['price']



title = ['Price vs Year', 

         'Price vs Mileage', 

         'Price vs Tax', 

         'Price vs Mpg', 

         'Price vs Engine Size']



fig,ax = plt.subplots(3,2,figsize=(20,20))



i = 0

for rows in range(3):

    for cols in range(2):

        if rows == 2 and cols == 1:

            fig.delaxes(ax[rows,cols])

            break

        ax[rows,cols].scatter(x = X[X.columns[i]], y = Y)

        ax[rows,cols].set_title(title[i])

        i = i+1

        

fig.subplots_adjust(hspace=0.5, wspace=0.5)
X = data_encode_dummy_transform[['year','mileage','tax','mpg','engineSize']]

X.corr()
VIF = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], 

                index=X.columns)

print(VIF)
X = data_encode_dummy_transform.drop(columns = ['price'])

Y = data_encode_dummy_transform['price']



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)



X_train = X_train.reset_index(drop = True)

X_test = X_test.reset_index(drop = True)

Y_train = Y_train.reset_index(drop = True)

Y_test = Y_test.reset_index(drop = True)
# Define the PCA object

pca = PCA()



# Preprocessing (1): first derivative

d1X = savgol_filter(X_train, 25, polyorder = 5, deriv=1)



# Preprocess (2) Standardize features by removing the mean and scaling to unit variance

Xstd = StandardScaler().fit_transform(d1X[:,:])



# Run PCA producing the reduced variable Xreg and select the first pc components

Xreg = pca.fit_transform(Xstd)[:,:]



XGB = XGBRegressor(random_state=0)



XGB.fit(Xreg,Y_train)
XGB.score(Xreg,Y_train)
Y_PCA_XGB = XGB.predict(Xreg)



mean_squared_error(Y_train, Y_PCA_XGB)
resid = Y_train - Y_PCA_XGB



sns.distplot(resid)
result = runstest_1samp(resid)[1]

print('P-value :',result)
plt.xlabel('Fitted Values')

plt.ylabel('Residuals')

plt.scatter(Y_PCA_XGB,resid)
# Preprocessing (1): first derivative

d1X = savgol_filter(X_test, 25, polyorder = 5, deriv=1)



# Preprocess (2) Standardize features by removing the mean and scaling to unit variance

Xstd = StandardScaler().fit_transform(d1X[:,:])



# Run PCA producing the reduced variable Xreg and select the first pc components

Xreg = pca.fit_transform(Xstd)[:,:]



prediction = XGB.predict(Xreg)



data_test = {

    'Y_test' : Y_test,

    'Prediction' : prediction

}



pd.DataFrame(data_test)
mean_squared_error(Y_test, prediction)