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
raw_input = pd.read_csv('../input/train.csv')
raw_input
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

mean_imputer = Imputer()
name_vectorizer = CountVectorizer()
name_vectorizer.fit(raw_input.Name)

cabin_vectorizer = CountVectorizer()
cabin_vectorizer.fit(raw_input.Cabin.values.astype('U'))

def select_features(raw_input):
    X = pd.DataFrame(data = {
        'PClass': raw_input.Pclass,
        'is_male': (raw_input.Sex == 'male') * 1,
        'is_female': (raw_input.Sex == 'female') * 1,
        'Age': [x[0] for x in mean_imputer.fit_transform(raw_input[['Age']])],
        'SipSp': raw_input.SibSp,
        'Parch': raw_input.Parch,
        'log_Fare': [np.log(x[0]+1) for x in mean_imputer.fit_transform(raw_input[['Fare']])],
        'embarked_C': (raw_input.Embarked == 'C') * 1,
        'embarked_S': (raw_input.Embarked == 'S') * 1,
        'embarked_Q': (raw_input.Embarked == 'Q') * 1,
        })
    name_tokenized = name_vectorizer.transform(raw_input.Name)
    cabin_tokenized = cabin_vectorizer.transform(raw_input.Cabin.values.astype('U'))
    return pd.concat([X.to_sparse(), pd.SparseDataFrame(name_tokenized).fillna(0), pd.SparseDataFrame(cabin_tokenized).fillna(0)], axis=1)

X = select_features(raw_input)
X
y = raw_input.Survived
y
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

cv_predicted = log_reg.predict(X_cv)
accuracy_score(y_cv, cv_predicted)
from sklearn.ensemble import RandomForestClassifier

ran_for = RandomForestClassifier()
ran_for.fit(X_train, y_train)

cv_rf_predicted = ran_for.predict(X_cv)
accuracy_score(y_cv, cv_rf_predicted)
log_reg_all = LogisticRegression()
log_reg_all.fit(X, y)

test_raw_input = pd.read_csv('../input/test.csv')
X_test = select_features(test_raw_input)

test_predicted = log_reg.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test_raw_input.PassengerId,
    'Survived': test_predicted
})
display(submission)
submission.to_csv('logreg_submission.csv', index=False)
rf_all = RandomForestClassifier()
rf_all.fit(X, y)

rf_test_predicted = rf_all.predict(X_test)
rf_submission = pd.DataFrame({
    'PassengerId': test_raw_input.PassengerId,
    'Survived': rf_test_predicted
})
display(rf_submission)
rf_submission.to_csv('rf_submission.csv', index=False)