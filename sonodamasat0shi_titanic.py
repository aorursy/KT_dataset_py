import numpy as np

import pandas as pd



DATASET_DIR = '/kaggle/input/titanic'

X_tr_df = pd.read_csv(DATASET_DIR + '/train.csv', index_col='PassengerId')

X_ts = pd.read_csv(DATASET_DIR + '/test.csv', index_col='PassengerId')

y_tr = X_tr_df['Survived']

X_tr = X_tr_df.drop(columns='Survived')



print('\nTraining data:')

X_tr.describe()
print('\nTest data:')

X_ts.describe()
def parse_ticket(X):

    header = []

    number = []

    for tt in X['Ticket']:

        if tt == 'LINE': # for several passengers

            header.append(tt)

            number.append(0)

        else:

            space = tt.rfind(' ')

            if space >= 0:

                header.append(tt[:space])

            else:

                header.append(None)

            number.append(int(tt[space+1:]))

    X['Ticket_header'] = header

    X['Ticket_number'] = number

    X.drop(columns='Ticket', inplace=True)



parse_ticket(X_tr)

X_tr.head()
parse_ticket(X_ts)

X_ts.head()
from sklearn.model_selection import train_test_split

X_tr_t, X_tr_v, y_tr_t, y_tr_v = train_test_split(X_tr, y_tr, test_size=0.25)



columns_num = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Ticket_number']

columns_cat = ['Sex', 'Embarked']
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
steps_num = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),

                            ('scaler', StandardScaler()),

                           ])
steps_cat = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='EMPTY')),

                            ('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore')),

                           ])
preprocessor = ColumnTransformer(transformers=[

    ('numeric_transformer', steps_num, columns_num),

    ('categorical_transformer', steps_cat, columns_cat),

    ], sparse_threshold=0.)

X_tr_t_pre = preprocessor.fit_transform(X_tr_t)

X_tr_v_pre = preprocessor.transform(X_tr_v)
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



optimizers = (['svm_rbf', SVC(kernel='rbf', gamma='scale')],

              ['svm_poly2', SVC(kernel='poly', degree=2, gamma='scale')],

              ['svm_poly3', SVC(kernel='poly', degree=3, gamma='scale')],

              ['kNN3', KNeighborsClassifier(n_neighbors=3)],

              ['kNN5', KNeighborsClassifier(n_neighbors=5)],

              ['kNN10', KNeighborsClassifier(n_neighbors=10)],

              ['kNN20', KNeighborsClassifier(n_neighbors=20)],

              ['naive_bayes', BernoulliNB()],

              ['random_forest', RandomForestClassifier(n_estimators=3)],

              ['ada_boost', AdaBoostClassifier(base_estimator=None, n_estimators=30)])
from sklearn.model_selection import cross_validate



cv_results = {}

for name, estimator in optimizers:

    estimator.fit(X_tr_t_pre, y_tr_t)

    print(f'      * {name:16}: training -> {estimator.score(X_tr_t_pre, y_tr_t):5.3}'

          f' / test -> {estimator.score(X_tr_v_pre, y_tr_v):5.3}')

del name

del estimator
X_tr_pre = preprocessor.transform(X_tr)

X_ts_pre = preprocessor.transform(X_ts)

for name, estimator in optimizers:

    estimator.fit(X_tr_pre, y_tr)

    out = pd.DataFrame({'PassengerID': X_ts.index,

                        'Survived': estimator.predict(X_ts_pre)})

    out.to_csv(f'model_v1_{name}.csv', index=False)