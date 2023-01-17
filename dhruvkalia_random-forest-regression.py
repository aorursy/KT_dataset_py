# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing data

df = pd.read_csv("../input/calcofi/bottle.csv")

df = df.loc[1:1000, ['T_degC','Salnty']]

df.head()
# Dividing dependent and independent variables

train = df[['Salnty']]

test = df[['T_degC']]
# Fixing missing values issue and imputing

imputer = SimpleImputer(strategy="mean")

train_imputed = pd.DataFrame(imputer.fit_transform(train))

test_imputed = pd.DataFrame(imputer.fit_transform(test))

train_imputed.columns = train_imputed.columns

test_imputed.columns = test.columns
# Splitting into train and test dataset

X_train, X_test, y_train, y_test = train_test_split(train_imputed, test_imputed, test_size=0.2, random_state=0)
# Define model

rfg = RandomForestRegressor(n_estimators=100, random_state=0)

rfg.fit(X_train, y_train['T_degC'])
# Predicting and plotting

test_y_hat = rfg.predict(X_test)

plt.scatter(X_test, y_test, color='blue')

plt.scatter(X_test, test_y_hat, color='red')

plt.title('Truth or Bluff (Random Forest Regression)')

plt.xlabel("Salinity")

plt.ylabel("Temparature")

plt.show()
# Accuracy measures

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - np.asanyarray(y_test))))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - np.asanyarray(y_test)) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat, np.asanyarray(y_test)))