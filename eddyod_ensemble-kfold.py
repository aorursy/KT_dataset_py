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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
NFOLDS = 10
trainfile = "../input/train.json"
train = pd.read_json(trainfile)
train['ingredients'] = train.ingredients.map(lambda x: ", ".join(x) )
X_train = train['ingredients']
y_train = train['cuisine']
targets = np.unique(y_train)
testfile = "../input/test.json"
test = pd.read_json(testfile)
test['ingredients'] = test.ingredients.map(lambda x: ",".join(x) )
X_test = test['ingredients']
clf1 = LogisticRegression()
clf3 = ExtraTreesClassifier()
clf5 = xgb.XGBClassifier(missing=np.nan, max_depth=7, n_estimators=250, learning_rate=0.05, nthread=2, subsample=0.95, colsample_bytree=0.85, seed=42)
clfs = EnsembleVoteClassifier(clfs=[clf1, clf3, clf5], weights=[1, 1, 1], voting='soft')
model = Pipeline([('vect', CountVectorizer(analyzer='word' ) ),
                     ('clf', clfs )])
input_shape = (len(X_test), len(targets) )
predictions_full = np.zeros(input_shape)
%%time
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
num_fold = 0
final_score = 0.0
for train_index, test_index in kf.split(X_train):
    X_training = X_train[train_index]
    y_training = y_train[train_index]
    X_testing = X_train[test_index]
    y_testing = y_train[test_index]
    num_fold += 1
    model.fit(X_training, y_training)
    predictions = model.predict_proba(X_test)
    predictions_full += predictions
    score = model.score(X_testing, y_testing)
    final_score += score
    print('KFold {} of {} with score {}'.format(num_fold, NFOLDS, score))
final_score = final_score / NFOLDS
print('Final score {}'.format(final_score))
predicts = predictions_full / float(NFOLDS)
predicts = np.argmax(predicts, axis=1)
y_classes = model.classes_
final_predictions = [y_classes[p] for p in predicts]
SUBMISSION = "sample_submission.csv"
test['cuisine'] = final_predictions
test.drop(['ingredients'], axis=1, inplace=True)
test.to_csv(SUBMISSION, index=False)
