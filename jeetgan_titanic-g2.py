# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

test_filepath = '/kaggle/input/titanic/test.csv'

train_filepath = '/kaggle/input/titanic/train.csv'

gender_subfilepath = '/kaggle/input/titanic/gender_submission.csv'
test_data = pd.read_csv(test_filepath)

train_data = pd.read_csv(train_filepath)

test_data_Y = pd.read_csv(gender_subfilepath)
train_data
test_data
gender_data
train_data = train_data.drop(columns = ['Name', 'Ticket', 'PassengerId'])
train_data
train_data.rename(columns = {'Sex':'isMale'}, inplace = True)

train_data['isMale'] = [1 if x == 'male' else 0 for x in train_data['isMale']]

train_data
dropped_cat = train_data.drop(columns = ['Cabin', 'Embarked'])
dropped_cat = dropped_cat.dropna()

dropped_cat_Y = dropped_cat['Survived']

dropped_cat_X = dropped_cat.drop(columns = ['Survived'])
logistic_reg_num = LogisticRegression(random_state = 0)

logistic_reg_num.fit(dropped_cat_X, dropped_cat_Y)
cat_test_data = test_data.drop(columns = ['Name', 'Ticket', 'PassengerId', 'Cabin', 'Embarked'])

cat_test_data.rename(columns = {'Sex':'isMale'}, inplace = True)

cat_test_data['isMale'] = [1 if x == 'male' else 0 for x in cat_test_data['isMale']]

cat_test_data
cat_test_data.isna().sum(axis = 1)
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

cat_test_data_imputed = my_imputer.fit_transform(cat_test_data)
predictions = logistic_reg_num.predict(cat_test_data_imputed)
from sklearn.metrics import classification_report

pred_result = classification_report(test_data_Y['Survived'], predictions)
import matplotlib.pyplot as plt

plt.hist(test_data_Y['Survived'], normed=True, bins=2)
test_data_Y
dic = {'PassangerId': test_data_Y['PassengerId'], 'Survived': predictions}

submission_df = pd.DataFrame(dic)
submission_df.to_csv('out.csv')
print(pred_result)
logistic_reg = LogisticRegression(random_state = 0)

logistic_reg.fit(final_test, final_test_Y)
unique_values = {}

for column_name in final_test_X.columns:

    unique_values[column_name] = final_test_X[column_name].unique()