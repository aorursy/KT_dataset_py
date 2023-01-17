import pandas as pd

import numpy as np  

import matplotlib.pyplot as plt  

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import seaborn as sns



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
datos = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")



datos.shape
datos.describe()
datos.head(9)
sns.pairplot(datos)
datos.corr().style.background_gradient(cmap='coolwarm', axis=None)
np.sum(pd.isnull(datos))
X = datos[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']].values

y = datos[['quality']].values
print(f"Dimensiones de X: {X.shape}")

print(f"Dimensiones de Y: {y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_test[:,0], y_test,  color='gray')

plt.plot(X_test[:,0], y_pred, color='red', linewidth=2)

plt.show()
coeff_df = pd.DataFrame(regressor.coef_.reshape((11,1)), datos.columns[:-1].values, columns=['Coefficient'])  

coeff_df
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaled_df = scaler.fit_transform(datos)

scaled_df = pd.DataFrame(scaled_df, columns=datos.columns)



scaled_df.describe()
X = scaled_df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']].values

y = scaled_df[['quality']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))