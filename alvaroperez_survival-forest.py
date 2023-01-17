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
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)
data.head()
data_embarked = data[['Embarked','Survived']]
count_s = data_embarked.loc[data_embarked['Embarked'] == 'S'].Survived.sum()
count_q = data_embarked.loc[data_embarked['Embarked'] == 'Q'].Survived.sum()
count_c = data_embarked.loc[data_embarked['Embarked'] == 'C'].Survived.sum()

d = {'Embarked': ['S', 'Q', 'C'], 'Total Survived': [count_s, count_q, count_c]}
count_by_embarked = pd.DataFrame(data=d)
count_by_embarked.plot.bar(x='Embarked', y = 'Total Survived')
data.Survived.hist()
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
le_emb = LabelEncoder()


my_pipeline = make_pipeline(Imputer(), RandomForestClassifier())
data.columns
data = data.fillna(method = 'backfill')
le_sex.fit(data['Sex'])
data['Sex'] = le_sex.transform(data['Sex'])
le_emb.fit_transform(data['Embarked'])
data['Embarked'] = le_emb.transform(data['Embarked'])
selected_columns = ['PassengerId', 'Pclass', 'Age', 'SibSp',
       'Parch', 'Fare','Sex', 'Embarked']
y = data.Survived
X = data[selected_columns]
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,test_size = 0.25)
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(val_X)
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, val_y)))
main_file_path = '../input/test.csv'
data_test = pd.read_csv(main_file_path)
data_test.head()
data_test.fillna(method = 'backfill')
data_test['Sex'] = le_sex.transform(data_test['Sex'])
data_test['Embarked'] = le_emb.transform(data_test['Embarked'])
selected_columns = ['PassengerId', 'Pclass', 'Age', 'SibSp',
       'Parch', 'Fare','Sex', 'Embarked']
X_test = data_test[selected_columns]
test_predictions = my_pipeline.predict(X_test)
my_submission = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': test_predictions})
# you could use any filename. We choose submission here PassengerId,Survived

my_submission.to_csv('submission_last.csv', index=False)
