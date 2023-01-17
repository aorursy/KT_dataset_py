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
train_data = pd.read_csv("../input/train.csv")
train_data.describe()
train_data
submission = pd.read_csv("../input/gender_submission.csv")
submission
predictors=['PassengerId','Pclass','Age','SibSp','Parch','Fare']
train_predictors = train_data[predictors]
target = train_data['Survived']
from sklearn.tree import DecisionTreeRegressor

survived_model = DecisionTreeRegressor()
#impute missing values
from sklearn.preprocessing import Imputer

my_imputer = Imputer()

imputed_train_data = my_imputer.fit_transform(train_predictors)

survived_model.fit(imputed_train_data, target)
predicted_y_value = survived_model.predict(imputed_train_data)
from sklearn.metrics import mean_absolute_error

print('Mean absolute error of predicted train data: ' + str(float(mean_absolute_error(target, predicted_y_value))))
predicted_y_value
target
test_data = pd.read_csv("../input/test.csv")
test_data
test_predictors = test_data[predictors]
#handle missing values

imputed_test_predictors = my_imputer.transform(test_predictors)
test_prediction = survived_model.predict(imputed_test_predictors)
test_prediction = test_prediction.astype(int)
my_submission = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':test_prediction})
my_submission
print(mean_absolute_error(submission.Survived, my_submission.Survived))
my_submission.to_csv("titanic_predictions.csv",index=False)
