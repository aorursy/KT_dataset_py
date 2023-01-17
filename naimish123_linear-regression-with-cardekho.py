# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

data=pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")

data.head()
pair_df = [data[["Year", "Present_Price", "Kms_Driven", "Owner"]], 

           pd.get_dummies(data[["Fuel_Type", "Seller_Type", "Transmission"]], 

                          drop_first=True), data[["Selling_Price"]]]

X = pd.concat(pair_df, axis=1)

y = data[["Selling_Price"]]



# Independent Variable

X.head()
plt.figure(figsize=(16,8))

corrmat = X.corr()

cols = corrmat.index

cm = np.corrcoef(X[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 

                 yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Let's delete the Selling_Price from X

X.drop(labels=["Selling_Price"], axis=1, inplace=True)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)



X_train = X_train.values

y_train = y_train.values

X_test = X_test.values

y_test = y_test.values
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)
# Calculating Loss

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



# MAE on training data 

print("MSE: ", mean_absolute_error(y_true=y_train, y_pred=linreg.predict(X_train)))



# MSE on training data

print("MAE: ", mean_squared_error(y_true=y_train, y_pred=linreg.predict(X_train)))
np.round(linreg.coef_.ravel(), 3)
X.head(1)
# For training data

plt.figure(figsize=(20, 10))

plt.plot(range(0, len(y_train)), y_train, label="TrueValues", marker="*", linewidth=3)

plt.plot(range(0, len(y_train)), linreg.predict(X_train), label="PredictedValues", marker="*", linewidth=3)

plt.xlabel("Indices",fontsize=20)

plt.ylabel("Selling Price of Cars",fontsize=20)

plt.title("True Selling Price Vs. Predicted Selling Price",fontsize=20)

plt.show()
# For Test data

plt.figure(figsize=(20, 10))

plt.plot(range(0, len(y_test)), y_test, label="TrueValues", marker="*", linewidth=3)

plt.plot(range(0, len(y_test)), linreg.predict(X_test), label="PredictedValues", marker="o", linewidth=3)

plt.xlabel("Indices",fontsize=20)

plt.ylabel("Selling Price of Cars",fontsize=20)

plt.title("True Selling Price Vs. Predicted Selling Price",fontsize=20)

plt.legend()

plt.show()