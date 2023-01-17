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
#importing the file

import matplotlib.pyplot as py

data = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

data.head()
#analysing the data

data.info()
#print(data.describe)

import seaborn as sns

sns.countplot(x=data.sqft_living, data=data)

sns.countplot(x="price", data=data)
sns.countplot(x="sqft_living", hue="bedrooms", data=data)
#train & test model

X = data[['sqft_above', 'sqft_basement']]

#X = data[['price', 'bedrooms', 'bathrooms','floors', 'yr_built']]

y = data['sqft_living']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
#model training

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

model = LinearRegression()

output = model.fit(X_train, y_train)





#evaluate

Ypredict = output.predict(X_test)

R2 = r2_score(y_test, Ypredict)

MSR = mean_squared_error(y_test, Ypredict)

print("Score: ", R2*100)

print("MSR: ", MSR)
import numpy as np

a = np.arange(30)

py.plot(a, y_test.head(30), label="Actual")

py.scatter(a, Ypredict[0:30], label = "Predicted")

py.legend()

py.show