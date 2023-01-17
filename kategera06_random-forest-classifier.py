from sklearn import preprocessing, ensemble, model_selection



import numpy as np

import pandas as pd

import xgboost as xgb

import seaborn as sns
%pylab inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/train.csv.zip', header=0, sep=',')

test_data = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/test.csv.zip', header=0, sep=',')
print('Shape of train data: ', train_data.shape)

print('Shape of test data: ', test_data.shape)
train_data.head()
test_data.head()
print('Column names:', list(train_data.columns))
train_data = train_data.drop(['id'], axis=1)



test_id = test_data['id']

test_data = test_data.drop(['id'], axis=1)
sns.pairplot(train_data.drop('color', axis = 1), hue = 'type', palette = 'muted', diag_kind='kde')
sns.countplot(x='color', hue='type', data=train_data)
sns.countplot(x='type', data=train_data)
color_le = preprocessing.LabelEncoder()



train_data_x = train_data

color_le.fit(train_data_x['color'])

train_data_x['color_int'] = color_le.transform(train_data_x['color'])

train_data_x = train_data_x.drop(['color', 'type'], axis=1)

train_data_x = train_data_x.to_numpy()



type_le = preprocessing.LabelEncoder()



train_data_y = train_data['type']

type_le.fit(train_data_y)

train_data_y = type_le.transform(train_data_y)



print('Unique type values:', train_data.type.unique())

print()

print('Original train_set_y:', np.array(train_data.type[:5]))

print('Encoded train_set_y:', train_data_y[:5])
test_data = test_data

color_le.fit(test_data['color'])

test_data['color_int'] = color_le.transform(test_data['color'])

test_data = test_data.drop(['color'], axis=1)

test_data = test_data.to_numpy()
print('test_data:')

print(test_data[:5])
print('train_data_x:')

print(train_data_x[:5])

print('shape:', train_data_x.shape, 'type:', type(train_data_x))

print()

print('train_data_y:')

print(train_data_y[:5])

print('shape:', train_data_y.shape, 'type:', type(train_data_y))
n_trees = [1] + list(range(5, 55, 5))

scoring = []

for n_tree in n_trees:

    estimator = ensemble.RandomForestClassifier(n_estimators = n_tree, min_samples_split=5, random_state=1)

    score = model_selection.cross_val_score(estimator, train_data_x, train_data_y, 

                                             scoring = 'accuracy', cv = 3)    

    scoring.append(score)

scoring = np.asmatrix(scoring)



scoring
pylab.plot(n_trees, scoring.mean(axis = 1), marker='.', label='RandomForest')

pylab.grid(True)

pylab.xlabel('n_trees')

pylab.ylabel('score')

pylab.title('Accuracy score')

pylab.legend(loc='lower right')
from sklearn.model_selection import GridSearchCV



params = {'n_estimators': n_trees}



grid_search = GridSearchCV(ensemble.RandomForestClassifier(min_samples_split=5, random_state=1), params, cv=3, scoring='accuracy')



grid_search.fit(train_data_x, train_data_y)



print('Best n of trees: {}, best accuracy score: {}'.format(grid_search.best_params_['n_estimators'], grid_search.best_score_))
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(train_data_x, train_data_y, test_size=0.33, random_state=42)
best_n_tree = grid_search.best_params_['n_estimators']

model = ensemble.RandomForestClassifier(n_estimators = n_tree, min_samples_split=5, random_state=1)

model.fit(train_set_x, train_set_y)

pred_y = model.predict(test_set_x)



score = accuracy_score(test_set_y, pred_y)

print('Accuracy score:', score)
test_pred = model.predict(test_data)

encoded_test_pred = type_le.inverse_transform(test_pred)

print('Test predictions:', encoded_test_pred[:5])
submission = pd.DataFrame({'id': test_id, 'type':encoded_test_pred})

submission.to_csv('submission.csv', index=False)