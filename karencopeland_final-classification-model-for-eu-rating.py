# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pycaret import classification
from pycaret.classification import *
import numpy as np
import pandas as pd
os.chdir('/kaggle/input/predicting-energy-rating-from-raw-data')

train = pd.read_csv('train_rating_eu.csv')
test = pd.read_csv('test_rating_eu.csv')

test = test.drop(['building_id', 'site_id', 'Unnamed: 0'], axis=1)
test.info()
train = train.drop(['building_id', 'site_id', 'Unnamed: 0'], axis=1)
train.info()
train['rating'].unique()
train['rating'] = train['rating'].replace(['A', 'B', 'C', 'D', 'E', 'F', 'G'], [0, 0, 1, 1, 2, 2, 2])
os.chdir('/kaggle/working/')
cla = classification.setup(data=train, target='rating', train_size=0.7)
m = compare_models()
cat = classification.create_model('catboost')
tuned_cat = classification.tune_model(cat)
bagged_cat = classification.ensemble_model(tuned_cat)
predictions = classification.predict_model(cat)
predictions = classification.predict_model(tuned_cat)
predictions = classification.predict_model(bagged_cat)
classification.plot_model(tuned_cat)
classification.evaluate_model(tuned_cat)
classification.finalize_model(tuned_cat)
predictions_test = classification.predict_model(tuned_cat, data=test)
labels = predictions_test['Label']
pd.value_counts(labels).plot.bar(figsize=(10,5))
y = train['rating']
X = train.drop(['rating'], axis=1)
X.info()
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2)
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)
from catboost import Pool, CatBoostClassifier

train_dataset = Pool(data=Xtrain, label=ytrain)
test_dataset = Pool(data=Xtest, label=ytest)


model = CatBoostClassifier(iterations=100,
                           learning_rate=.01,
                           depth=2,
                           loss_function='MultiClass')
model.fit(train_dataset)
# Get predicted classes
preds_class = model.predict(test_dataset)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(test_dataset)
preds_proba
def getLetter(prob, cat):
    letter = [0] * len(cat)
    i = 0
    for p in prob:
        if cat[i] == 0:
            if p[1] > p[2]:
                letter[i] = 'A'
            else:
                letter[i] = 'B'
        elif cat[i] == 1:
            if p[0] > p[2]:
                letter[i] = 'C'
            else:
                letter[i] = 'D'
        else:
            if p[0] > p[1]:
                letter[i] = 'E'
            else:
                letter[i] = 'F/G'
        i = i + 1
    return letter
letters = pd.DataFrame(getLetter(preds_proba, preds_class))
pd.value_counts(letters[0]).plot.bar(figsize=(10,5))
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(preds_class, ytest)
accuracy
predictions_test = pd.DataFrame(model.predict(test))
preds_proba_test = model.predict_proba(test)
pd.value_counts(predictions_test[0]).plot.bar(figsize=(10,5))
letters = pd.DataFrame(getLetter(preds_proba_test, predictions_test[0]))
pd.value_counts(letters[0]).plot.bar(figsize=(10,5))
