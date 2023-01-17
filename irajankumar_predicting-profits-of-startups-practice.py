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
import pandas as pd

dataset = pd.read_csv("../input/50-startups/50_Startups.csv")

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 4].values



print(x)

print(y)
# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

x[:, 3] = labelencoder.fit_transform(x[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])

x = onehotencoder.fit_transform(x).toarray()



print(x)


# Avoiding the Dummy Variable Trap

x = x[:, 1:]

print(x)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
# Predicting the Test set results

y_pred = regressor.predict(x_test)
# Take a look at predicted values



print(y_pred)



print(y_test)