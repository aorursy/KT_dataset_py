# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



titanic_filepath = "../input/titanic/train.csv"



titanic_data = pd.read_csv(titanic_filepath, index_col = "PassengerId")



titanic_test_filepath = '../input/titanic/test.csv'



titanic_test_data = pd.read_csv(titanic_test_filepath)

X_test = titanic_test_data.drop(["PassengerId","Name", "Ticket", "Cabin"], axis = 1)



X_test.Sex[X_test.Sex == "male"] = 1

X_test.Sex[X_test.Sex == "female"] = 0



X_test.Embarked[X_test.Embarked == 'S'] = 1

X_test.Embarked[X_test.Embarked == 'C'] = 2

X_test.Embarked[X_test.Embarked == 'Q'] = 3



#sns.lmplot(x="Age", y="Fare", hue="Survived", data=titanic_data)



y = titanic_data["Survived"]

X = titanic_data.drop(["Survived", "Name", "Ticket", "Cabin"], axis = 1)



X.Embarked[X.Embarked == 'S'] = 1

X.Embarked[X.Embarked == 'C'] = 2

X.Embarked[X.Embarked == 'Q'] = 3



X.Sex[X.Sex == "male"] = 1

X.Sex[X.Sex == "female"] = 0



train_X, val_X, train_y, val_y = train_test_split(X, y)



my_imputer = SimpleImputer(strategy = "mean")



imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X))

imputed_X_valid = pd.DataFrame(my_imputer.transform(val_X))



imputed_X_train.columns = train_X.columns

imputed_X_valid.columns = val_X.columns



imputed_X = pd.DataFrame(my_imputer.fit_transform(X))

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))



imputed_X.columns = train_X.columns

imputed_X_test.columns = X_test.columns



length = len(X.Embarked)



print(length)



for i in range(length):

    imputed_X.Embarked[i] = round(imputed_X.Embarked[i])

    

for i in range(length):

    imputed_X.Age[i] = round(imputed_X.Age[i])

    

sns.lmplot(x = 'Age', y = 'Sex' , data = imputed_X)



def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators = 102, max_depth = 9, criterion = 'mae', random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
rf = RandomForestRegressor(n_estimators = 102, max_depth = 9, criterion = 'mae')

rf.fit(imputed_X, y)



preds_test = rf.predict(imputed_X_test)



length = len(preds_test)



for i in range(length):

    if preds_test[i] >= 0.5:

        preds_test[i] = 1

    else:

        preds_test[i] = 0



preds_test = preds_test.astype('int32')

        

print(preds_test[4])





output = pd.DataFrame({'PassengerId': X_test.index,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)