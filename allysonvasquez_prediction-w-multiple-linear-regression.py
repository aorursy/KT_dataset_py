# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
body_data = pd.read_csv('/kaggle/input/500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv')

body_data.head()   # prints first 5 rows of DataFrame
print(body_data.shape)    # prints the number of (rows, cols)

print(body_data.columns)  # prints list of columns
body_data.isnull().any()
print(body_data.Gender.describe())

print(body_data.Gender.unique())
body_data.Gender = body_data.Gender.replace({'Male': 1})

body_data.Gender = body_data.Gender.replace({'Female': 2})

body_data.Gender
body_data.head(10)
# features are ['Gender', 'Height', 'Weight']

X = body_data.drop('Index', axis=1) # Keeps all columns except Index

X.head()
y = body_data.Index
y
from sklearn import linear_model

reg = linear_model.LinearRegression()



reg.fit(X,y)
predictions = reg.predict(X)  # Stored in a numpy array

predictions[:10]
print("Predicted Mean", predictions.mean())

print("Actual Mean", y.mean())

print('\n')

print("Predicted Standard Deviation", predictions.std())

print("Actual Standard Deviation", y.std())

print('\n')

from sklearn.metrics import mean_squared_error



lin_mse = mean_squared_error(y, predictions)

lin_rmse = np.sqrt(lin_mse)

print("Prediction Error", lin_rmse)
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

ax.scatter(y, predictions, edgecolors=(0, 0, 0))

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)



ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()