# Must be imported before importing other libraries

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics, neighbors



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if '.csv' in filename:

            print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def input_train_test():

    root = '../input/data-preprocessing/'

    train = pd.read_csv(root+'train.csv')

    test = pd.read_csv(root+'test.csv')

    

    if train.shape[0] == 82332:

        print("Train and test sets are reversed here. Fixing them.")

        train, test = test, train

    drop_columns = ['attack_cat', 'id']

    for df in [train, test]:

        for col in drop_columns:

            if col in df.columns:

                print('Dropping '+col)

                df.drop([col], axis=1, inplace=True)

    return train, test



train, test = input_train_test()

# test = pd.read_csv('/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv')

# train = pd.read_csv('/kaggle/input/unsw-nb15/UNSW_NB15_testing-set.csv')
def detection_rate(y_true, y_pred):

    CM = metrics.confusion_matrix(y_true, y_pred)

    TN = CM[0][0]

    FN = CM[1][0]

    TP = CM[1][1]

    FP = CM[0][1]

    return TP/(TP+FN)



def false_positive_rate(y_true, y_pred):

    CM = metrics.confusion_matrix(y_true, y_pred)

    TN = CM[0][0]

    FN = CM[1][0]

    TP = CM[1][1]

    FP = CM[0][1]

    return FP/(FP+TN)



def get_xy(df):

    return pd.get_dummies(df.drop(['attack_cat', 'label'], axis=1)), df['label']



def get_train_test(train, test):

    x_train, y_train = get_xy(train)

    x_test, y_test = get_xy(test)



    print("Column mismatch {0}, {1}".format(set(x_train.columns)- set(x_test.columns),  set(x_test.columns)- set(x_train.columns)))

    features = list(set(x_train.columns) & set(x_test.columns))

    print(f"Number of features {len(features)}")

    x_train = x_train[features]

    x_test = x_test[features]

    return x_train, y_train, x_test, y_test



def results(y_test, y_pred):

    print(f"Accuracy {metrics.accuracy_score(y_test, y_pred)*100}, F1-score {metrics.f1_score(y_test, y_pred)*100}")

    # print(metrics.classification_report(y_test, y_pred))

    print("DR {0}, FPR {1}".format(detection_rate(y_test, y_pred)*100, false_positive_rate(y_test, y_pred)*100))
len(train.columns)
target = 'label'

x_train, y_train = train.drop([target], axis=1), train[target]

x_test, y_test = test.drop([target], axis=1), test[target]
model = neighbors.KNeighborsClassifier(n_neighbors=10)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

results(y_test, y_pred)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

results(y_test, y_pred)
from sklearn.svm import SVC
model = SVC()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

results(y_test, y_pred)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

results(y_test, y_pred)
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, random_state=1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

results(y_test, y_pred)