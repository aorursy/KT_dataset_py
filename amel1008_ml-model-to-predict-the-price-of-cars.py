# Importing the required libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

%matplotlib inline 

sns.set(color_codes=True)

np.random.seed(31415)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading the CSV file into a pandas dataframe.

df = pd.read_csv("../input/cars1/CARS.csv")

df.head(5)
colstocheck = df.columns

df[colstocheck] = df[colstocheck].replace({'\$':''}, regex = True)

df[colstocheck] = df[colstocheck].replace({',':''}, regex = True)

col_mask=df.isnull().any(axis=0) 

print(col_mask)

row_mask=df.isnull().any(axis=1)

df.loc[row_mask,col_mask]

df = df.dropna()

df['MSRP'] = df['MSRP'].astype(float)

df.head(5)
df = df.drop(['Make','Model','Type','Origin','DriveTrain','Invoice'],axis=1)

df.head(5)
# Importing all the required libraries

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
# Creating the Lasso Regression Model

reg = linear_model.Lasso(alpha=0.1)
X = df.drop("MSRP", axis=1)

y = df["MSRP"]

X = X.to_numpy()

X.ndim
y = y.to_numpy()

y.ndim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Fitting and predicting the trained values to the Lassor Regression Model

reg.fit(X_train, y_train)

pred = reg.predict(X_test)

# Printing the first five predicted values

pred[1:5]

plt.figure(figsize= (6, 6))

plt.title("Visualizing the Regression using Lasso Regression algorithm")

sns.regplot(x=pred, y=y_test, color = "teal")

plt.xlabel("New Predicted Price (MSRP)")

plt.ylabel("Old Price (MSRP)")

plt.show()
print("Mean Absolute Error is :", mean_absolute_error(y_test, pred))

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("Mean Squared Error is :", mean_squared_error(y_test, pred))

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("Coeffients are : ", reg.coef_)

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("Intercepts are :" ,reg.intercept_)

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("The R2 square value of Lasso is :", r2_score(y_test, pred)*100)
model = RandomForestRegressor()

model.fit(X_train, y_train)

pred = model.predict(X_test)



plt.figure(figsize= (6, 6))

plt.title("Visualizing the Regression using Random Forest Regression algorithm")

sns.regplot(pred, y_test, color = 'teal')

plt.xlabel("New Predicted Price (MSRP)")

plt.ylabel("Old Price (MSRP)")

plt.show()

model = RandomForestRegressor()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Mean Absolute Error is :", mean_absolute_error(y_test, pred))

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("Mean Squared Error is :", mean_squared_error(y_test, pred))

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("The R2 square value of RandomForest Regressor is :", r2_score(y_test, pred)*100)

print(" — — — — — — — — — — — — — — — — — — — — — — — ")
model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

plt.figure(figsize= (6, 6))

plt.title("Visualizing the Regression Linear Regression Algorithm")

sns.regplot(pred, y_test, color = 'teal')

plt.xlabel("New Predicted Price (MSRP)")

plt.ylabel("Old Price (MSRP)")

plt.show()
print("Mean Absolute Error is :", mean_absolute_error(y_test, pred))

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("Mean Squared Error is :", mean_squared_error(y_test, pred))

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("Coeffients are : ", model.coef_)

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("Intercepts are :" ,model.intercept_)

print(" — — — — — — — — — — — — — — — — — — — — — — — ")

print("The R2 square value of Linear Regression is :", r2_score(y_test, pred)*100)