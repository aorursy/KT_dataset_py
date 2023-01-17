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
path = '/kaggle/input/titanic/'

traindf = pd.read_csv(path + 'train.csv')

testdf = pd.read_csv(path + 'test.csv')

submission = pd.read_csv(path + 'gender_submission.csv')
traindf.info()
submission.info()
import catboost

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import train_test_split
traindf = traindf.fillna(-9999)

testdf = testdf.fillna(-9999)
testdf['Title'] = testdf.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

traindf['Title'] = traindf.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

testdf['Title'].value_counts()
traindf = traindf[['PassengerId', 'Title', 'Pclass','Sex', 'Fare', 'Age', 'Survived']]

testdf = testdf[['PassengerId', 'Title', 'Pclass','Sex', 'Age', 'Fare']]
X = traindf.drop('Survived', axis = 1)

y = traindf['Survived']

cat_features_index = np.where(X.dtypes != float)[0]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .25, random_state = 42)
model = CatBoostClassifier(

iterations = 50000,

task_type = 'GPU',

learning_rate = .001,

early_stopping_rounds = 1000,

depth = 10,

loss_function = 'CrossEntropy',

eval_metric = 'BalancedAccuracy')
model.fit(X_train, y_train, cat_features = cat_features_index, eval_set = (X_val, y_val), plot = True)
print(model.get_feature_importance())

print(X_train.columns)

print(cat_features_index)
#ID has no value...makes sense

#PClass has a value of 16

#Name has 0 value...again, make sense no feature engineering used

#Sex has a huge weight of 36

#Age is about 5

#Parch is 6, and the rest all settle around five...

#Let's redo the model with just the top 3 features, and PassengerID to make things easy

#Top three are Pclass, Sex, and Fare

submission['PassengerId'] = testdf['PassengerId']

submission['Survived'] = model.predict(testdf, prediction_type='Class')

submission['Survived'] = submission['Survived'].astype(int)

submission.to_csv('submission.csv', index = False)
model.predict(testdf)