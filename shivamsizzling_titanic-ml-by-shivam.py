# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# path of the file to read
titanic_file_path = '../input/train.csv'
titanic_data = pd.read_csv(titanic_file_path)
test_file_path = '../input/test.csv'
test_data = pd.read_csv(test_file_path)
features1 = ["Pclass", "Age"]
test_X = test_data[features1]

# counting the missing data
missing_val_count_by_column = titanic_data.isnull().sum()
print(missing_val_count_by_column[missing_val_count_by_column >0])
#do something for mising data
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
titanic_data_imputed=my_imputer.fit_transform(titanic_data[["Age"]])
titanic_data["Age"] = titanic_data_imputed
missing_val_count_by_column = titanic_data.isnull().sum()
print(missing_val_count_by_column[missing_val_count_by_column >0])
test_data_imputed=my_imputer.fit_transform(test_data[["Age"]])
test_data["Age"] = test_data_imputed
missing_val_count_by_column1 = test_data.isnull().sum()
print(missing_val_count_by_column1[missing_val_count_by_column1 >0])
test_X = test_data[features1]
#create target object and call it y
y = titanic_data.Survived
#create predictors
features = ["Pclass", "Age"]
X = titanic_data[features]


# Specify Model
titanic_model = DecisionTreeRegressor(random_state=1)
# Fit Model
titanic_model.fit(X, y)

# Predictions
my_predictions=titanic_model.predict(test_X)
for i,j in enumerate(my_predictions):
    if j<0.5:
        my_predictions[i] = 0
    else:
        my_predictions[i] = 1
my_predictions = [int(i) for i in my_predictions]
d = {"PassengerId":test_data.PassengerId, "Survived":my_predictions}
output = pd.DataFrame(d)
output.to_csv('submission.csv', index=False)
output