### Imports



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%pylab inline

import seaborn as sns # seaborn is made for visualisation

sns.set_style('darkgrid') # set seaborn styles to darkgrid

from sklearn.cross_validation import train_test_split

from sklearn.neighbors import KNeighborsRegressor



### Read, describe training data



# Input data files are available in the "../input/" directory.

train_csv = pd.read_csv("../input/train.csv")

train_csv.describe() # display summary stats for each column in training data
### Create the mdoel matrices



y_col = "SalePrice"

# X_cols = [col for col in train_csv.columns if col not in y_col]

X_cols = ["LotArea", "YearBuilt", "OverallQual"]



y = train_csv[y_col]

X = train_csv[X_cols]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
X_train.dtypes

# y.dtypes
neigh = KNeighborsRegressor(n_neighbors = 5)



neigh.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error



test_predict = neigh.predict(X_test)

test_error = mean_squared_error(y_test, test_predict)



print(y_test[:5], test_predict[:5], sqrt(test_error))