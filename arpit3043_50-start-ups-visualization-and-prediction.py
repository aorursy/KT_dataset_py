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
# importing Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Importing Dataset

dataset = pd.read_csv("/kaggle/input/various-expenses-and-the-profits-of-50-startups/50_Startups.csv")

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:,-1].values
# Encoding Categorical

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

print(X)
# Splitting data into test and train

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
X
y
X_train
X_test
y_train
y_test
# Training Multiple Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
# predicting Results

y_pred = regressor.predict(X_test)
X_pred = regressor.predict(X_train)
y_pred
np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.preprocessing import LabelEncoder

catToNum = LabelEncoder()

dataset["State"] = catToNum.fit_transform(dataset["State"])

# Preview

dataset.head()
dataset.rename(columns = {"R&D Spend" : "RD", "Marketing Spend" : "Marketing"}, inplace = True)

# Preview

dataset.head()

dataset.describe()
dataset.info()
dataset.describe
dataset.tail()
import seaborn as sns

plt.figure(figsize = (14, 8))

sns.boxplot(x = dataset.columns, y = [dataset[col] for col in dataset.columns])
dataset.corr()
# Check the differences in the means

# California: 0, Florida: 1, New York: 2

dataset.groupby("State", as_index = True).mean()["Profit"].to_frame()
# Check the differences in minimums and maximums

maximums = dataset.groupby("State").max()["Profit"].to_frame()

minimums = dataset.groupby("State").min()["Profit"].to_frame()

minimums.merge(maximums, on = "State").rename(columns = {"Profit_x" : "min", "Profit_y" : "max"})
sns.boxplot(data = dataset, x = "State", y = "Profit")
stateGroups = dataset.groupby("State", as_index = True)

gCal, gFlor, gNY = stateGroups.get_group(0)["Profit"], stateGroups.get_group(1)["Profit"], stateGroups.get_group(2)["Profit"]
fig, axes = plt.subplots(2, 2, figsize = (14, 12))

fig.suptitle("Regression plots: Profit vs expenses and State", fontsize = 20)

axesList = list(axes[0])

axesList.extend(list(axes[1]))



for i, axis in enumerate(axesList):

    col = dataset.columns[i]

    sns.regplot(data = dataset, x = col, y = "Profit", ax = axis)

    axis.set_title("Profit vs %s %s"%(col, "costs" if col != "State" else "categories"), fontsize = 15)



# plt.savefig("Profit_vs_Expenses.jpg")

plt.show()