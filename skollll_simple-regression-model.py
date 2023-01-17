# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import the imported libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Read the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
non_numeric_cols = list(test.head().select_dtypes(include=['object']))

numeric_cols = list(test.head().select_dtypes(exclude=['object']))
train_data = pd.concat([train[numeric_cols], pd.get_dummies(train[non_numeric_cols])], axis=1)

test_data = pd.concat([test[numeric_cols], pd.get_dummies(test[non_numeric_cols])], axis=1).reindex(columns=list(train_data), fill_value=0.0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_test_features = scaler.fit_transform(test_data.fillna(0.0).values)

scaled_train_features = scaler.fit_transform(train_data.fillna(0.0).values)
from sklearn.linear_model import Lasso, Ridge, SGDRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(scaled_train_features, train['SalePrice'], test_size=0.33, random_state=42)

models = {

    'Lasso' : Lasso(alpha = 1.0, max_iter = 500),

    'Ridge' : Ridge(alpha=1.0),

    'BaggingRegressor': BaggingRegressor(),

    'GradientBoostingRegressor': GradientBoostingRegressor(),

    'RandomForestRegressor': RandomForestRegressor(),

    'KNeighborsRegressor': KNeighborsRegressor()

}



for name, model in models.items():

    model.fit(X_train, y_train)

    print(name, ' : ', model.score(X_train, y_train), model.score(X_test, y_test))

    predicted = model.predict(X_test)

    print('  MSE(train): %f' % mean_squared_error(model.predict(X_train), y_train))

    print('  MSE(test) : %f' % mean_squared_error(model.predict(X_test), y_test)) 
# Using Ridge based on score

model = Ridge(alpha=1.0)

model.fit(scaled_train_features, train['SalePrice'])

predictions = model.predict(scaled_test_features)

results_dataframe = pd.DataFrame({

    "Id" : test['Id'],

    "SalePrice": predictions

})

# Set any -ve values to 0

results_dataframe.loc[results_dataframe.SalePrice < 0 , 'SalePrice'] = 0

results_dataframe.to_csv("first_submission.csv", index = False)