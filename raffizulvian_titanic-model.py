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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

train_data.info()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
x = train_data

y = x["Survived"]

x.drop(['Survived'], axis=1, inplace=True)



low_cardinality_cols = [cname for cname in x.columns if x[cname].nunique() < 10 and 

                        x[cname].dtype == "object"]



numeric_cols = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]



my_cols = low_cardinality_cols + numeric_cols



x_OH = pd.get_dummies(x[low_cardinality_cols])

x_new = pd.concat([x[numeric_cols], x_OH], axis=1)
test_OH = pd.get_dummies(test_data[low_cardinality_cols])

test_new = pd.concat([test_data[numeric_cols], test_OH], axis=1)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score, train_test_split



train_x, val_x, train_y, val_y = train_test_split(x_new, y)
train_x_plus = train_x.copy()

val_x_plus = val_x.copy()



cols_with_missing = [col for col in train_x.columns

                     if train_x[col].isnull().any()]



# Make new columns indicating what will be imputed

for col in cols_with_missing:

    train_x_plus[col + '_was_missing'] = train_x_plus[col].isnull()

    val_x_plus[col + '_was_missing'] = val_x_plus[col].isnull()



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(train_x_plus))

imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(val_x_plus))



# Imputation removed column names; put them back

imputed_X_train_plus.columns = train_x_plus.columns

imputed_X_valid_plus.columns = val_x_plus.columns

gbrt = GradientBoostingClassifier(max_depth=5, n_estimators=120, learning_rate=0.05)

gbrt.fit(imputed_X_train_plus, train_y)
from sklearn.metrics import accuracy_score

accuracy = [accuracy_score(val_y, y_pred) for y_pred in gbrt.staged_predict(imputed_X_valid_plus)]

bst_n_estimators = np.argmax(accuracy)

accuracy
import matplotlib.pyplot as plt

%matplotlib inline



x_axis = [x+1 for x in range(120)]

y_axis = accuracy



plt.plot(x_axis, y_axis)

plt.show()
gbrt_best = GradientBoostingClassifier(max_depth=5,n_estimators=bst_n_estimators, learning_rate=0.05)

gbrt_best.fit(imputed_X_train_plus, train_y)
test_plus = test_new.copy()



cols_with_missing = [col for col in test_new.columns

                     if test_new[col].isnull().any()]



# Make new columns indicating what will be imputed

for col in cols_with_missing:

    test_plus[col + '_was_missing'] = test_plus[col].isnull()

    

# Imputation

my_imputer = SimpleImputer()

imputed_test_plus = pd.DataFrame(my_imputer.fit_transform(test_plus))



# Imputation removed column names; put them back

imputed_test_plus.columns = test_plus.columns

imputed_test_plus = imputed_test_plus.drop('Fare_was_missing', axis=1)
predictions = gbrt_best.predict(imputed_test_plus)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")