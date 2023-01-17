# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



print(files)

# Any results you write to the current directory are saved as output.
data = pd.read_csv(files[0])

data.head()
data_simple = data._get_numeric_data().dropna()

data_simple = data_simple.drop(["id"], axis=1)

print(data_simple.shape)

data_simple.head()
X = data_simple.iloc[:,:-1].values

y = data_simple["revenue"].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=0)

X_scaler = StandardScaler().fit(X_train)

X_train_normalised = X_scaler.transform(X_train)

X_val_normalised = X_scaler.transform(X_val)

print(X_train_normalised[:5])

print(X_val_normalised[:5])
linear_model = LinearRegression().fit(X_train_normalised, y_train)

linear_model.score(X_val_normalised, y_val)
rf_model = RandomForestRegressor().fit(X_train_normalised, y_train)

rf_model.score(X_val_normalised, y_val)