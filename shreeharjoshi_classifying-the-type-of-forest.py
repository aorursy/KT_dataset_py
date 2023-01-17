import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

!ls ../input
train = pd.read_csv('../input/forest-cover-type-prediction/train.csv',

                   index_col='Id')

test = pd.read_csv('../input/forest-cover-type-prediction/test.csv',

                  index_col='Id')
train.head(1).T
train['Cover_Type'].value_counts()
def write_to_submission_file(predicted_labels, out_file,

                             target='Cover_Type', index_label="Id", init_index=15121):

    # turn predictions into data frame and save as csv file

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(init_index, 

                                                  predicted_labels.shape[0] + init_index),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
X_train, X_valid, y_train, y_valid = train_test_split(

    train.drop('Cover_Type', axis=1), train['Cover_Type'],

    test_size=0.3, random_state=101)
logit = LogisticRegression(C=1, solver='lbfgs', max_iter=500,

                           random_state=17, n_jobs=4,

                          multi_class='multinomial')

logit_pipe = Pipeline([('scaler', StandardScaler()), 

                       ('logit', logit)])
%%time

logit_pipe.fit(X_train, y_train)
logit_val_pred = logit_pipe.predict(X_valid)
accuracy_score(y_valid, logit_val_pred)
first_forest = RandomForestClassifier(

    n_estimators=100, random_state=17, n_jobs=4)
%%time

first_forest.fit(X_train, y_train)
forest_val_pred = first_forest.predict(X_valid)
accuracy_score(y_valid, forest_val_pred)
train.columns
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

for col in train.columns:

    train[col]=LabelEncoder().fit(train[col]).transform(train[col])

model= DecisionTreeClassifier(criterion= 'entropy',max_depth = 1)

AdaBoost= AdaBoostClassifier(base_estimator= first_forest, n_estimators= 400,learning_rate=1)

boostmodel= AdaBoost.fit(X_train, y_train)

y_predict= boostmodel.predict(X_valid)

accuracy_score(y_valid, y_predict)

write_to_submission_file(y_predict,'final answer.csv')