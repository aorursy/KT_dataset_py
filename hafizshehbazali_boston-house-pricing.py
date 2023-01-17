# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_boston

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



dataset = load_boston()



# describe dataset

print(dataset.keys())

print(dataset.DESCR)
boston_dataset = pd.DataFrame(dataset.data, columns=dataset.feature_names)

boston_dataset["MEDV"] = dataset.target;

print(boston_dataset.head())
# todo code here for PCA and features selection
from sklearn.model_selection import train_test_split

x = boston_dataset[["LSTAT","RM", "PTRATIO","TAX","INDUS"]]

y = boston_dataset[["MEDV"]]

x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)



from sklearn.linear_model import LinearRegression



linear_model = LinearRegression()

linear_model.fit(x_train, y_train)
prediction = linear_model.predict(x_test.head(5))



prediction_result = x_test.head(5)

prediction_result.insert(5,"predicted", prediction.flatten())

prediction_result
from sklearn.metrics import mean_squared_error, r2_score

prediction = linear_model.predict(x_test)

# root mean square error

rmse = np.sqrt(mean_squared_error(y_test, prediction))

print("root means squared error : {}".format(rmse))



r2_score = r2_score(y_test, prediction)

print("r2_score : {}".format(r2_score))