# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
test_data1 = pd.read_csv('../input/test.csv')
train_data.head()
train_data = train_data.drop(["Cabin","Ticket","Name","PassengerId"],axis = 1)
test_data= test_data1.drop(["Cabin","Ticket","Name","PassengerId"],axis = 1)
train_data.describe
train_data = pd.get_dummies(train_data)
test_data =pd.get_dummies(test_data)
X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_test  = test_data.copy()
from sklearn.impute import SimpleImputer
train_data.head()
X_train.shape, Y_train.shape, X_test.shape,test_data1.shape
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_train_X = my_imputer.fit_transform(X_train)
imputed_test_X = my_imputer.fit_transform(X_test)
melbourne_model = RandomForestClassifier(random_state=1)
melbourne_model.fit(imputed_train_X, Y_train)
val_predictions = melbourne_model.predict(imputed_test_X)
acc_knn = round(melbourne_model.score(imputed_train_X, Y_train) * 100, 2)
acc_knn
submission = pd.DataFrame({
        "PassengerId": test_data1["PassengerId"],
        "Survived": val_predictions
    })
submission.to_csv('submission.csv', index=False)








