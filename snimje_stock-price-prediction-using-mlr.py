# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection  import train_test_split

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read the train data from datasets

data = pd.read_csv("/kaggle/input/stock-data/trainset.csv")

data.head()
# convert date into proper format

data["Date"] = pd.to_datetime(data["Date"])

data.head()
# close is our target variable and all othres are independent variables

# check the relationship between dependent and independent variables

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.scatter(data["Close"], data["Open"], color="blue")

ax.scatter(data["Close"], data["High"], color="red")

ax.scatter(data["Close"], data["Low"], color="green")

ax.set_xlabel("Dependent variable")

ax.set_ylabel("Indepedent varibales")

ax.set_title("Co-Relation Scatter Plot")

plt.show()
# calculate day of the week from date

data["dayofweek"] = data["Date"].dt.dayofweek

data.head()
data['mon_fri'] = 0

for i in range(0,len(data)):

    if (data['dayofweek'][i] == 0 or data['dayofweek'][i] == 4):

        data['mon_fri'][i] = 1

    else:

        data['mon_fri'][i] = 0

data.head()
X = data[["Open", "High", "Low", "Adj Close", "Volume", "mon_fri"]].values

y = data["Close"].values
# split the data into train test and split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("X_train shape:", X_train.shape)

print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)

print("y_test shape:", y_test.shape)
# create model and fit the data

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

y_hat = lr_model.predict(X_test)

print(y_hat[0:5])
test_data = pd.read_csv("/kaggle/input/stock-data/testset.csv")

test_data.head()
# convert date into dayof week

test_data["Date"] = pd.to_datetime(test_data["Date"])

test_data["dayofweek"] = test_data["Date"].dt.dayofweek



test_data['mon_fri'] = 0

for i in range(0,len(test_data)):

    if (test_data['dayofweek'][i] == 0 or test_data['dayofweek'][i] == 4):

        test_data['mon_fri'][i] = 1

    else:

        test_data['mon_fri'][i] = 0

    

test_data.head()
y_testdata = test_data["Close"].values

X_testdata = test_data[["Open", "High", "Low", "Adj Close", "Volume", "mon_fri"]].values
# use data to predict values

y_predict = lr_model.predict(X_testdata)

print(y_predict[0:5])

print(y_testdata[0:5])
# calculate the accuracy of the model

from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_hat))

print('R²: %.2f' % r2_score(y_test, y_hat))