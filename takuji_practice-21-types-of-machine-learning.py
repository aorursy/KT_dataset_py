import pandas as pd

import numpy as np



train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

submission_df = pd.read_csv('../input/titanic/gender_submission.csv')
train_df = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

test_df = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
age_mean = train_df['Age'].mean()

fare_mean = train_df['Fare'].mean()



train_df['Age'] = train_df['Age'].fillna(age_mean)

test_df['Age'] = test_df['Age'].fillna(age_mean)

train_df['Fare'] = train_df['Fare'].fillna(fare_mean)

test_df['Fare'] = test_df['Fare'].fillna(fare_mean)

train_df['Embarked'] = train_df['Embarked'].fillna('S')

test_df['Embarked'] = test_df['Embarked'].fillna('S')

train_df.loc[train_df['Sex']=='male', 'Sex'] = 0

train_df.loc[train_df['Sex']=='female', 'Sex'] = 1

test_df.loc[test_df['Sex']=='male', 'Sex'] = 0

test_df.loc[test_df['Sex']=='female', 'Sex'] = 1

train_df.loc[train_df['Embarked']=='S', 'Embarked'] = 0

train_df.loc[train_df['Embarked']=='C', 'Embarked'] = 1

train_df.loc[train_df['Embarked']=='Q', 'Embarked'] = 2

test_df.loc[test_df['Embarked']=='S', 'Embarked'] = 0

test_df.loc[test_df['Embarked']=='C', 'Embarked'] = 1

test_df.loc[test_df['Embarked']=='Q', 'Embarked'] = 2



train_df['Sex'] = train_df['Sex'].astype(int)

test_df['Sex'] = test_df['Sex'].astype(int)

train_df['Embarked'] = train_df['Embarked'].astype(int)

test_df['Embarked'] = test_df['Embarked'].astype(int)
X_train = train_df.iloc[:, 1:]

y_train = train_df.iloc[:, 0]

X_test = test_df
from sklearn.preprocessing import StandardScaler, MinMaxScaler



ss = StandardScaler()

mms = MinMaxScaler()



ss.fit(X_train)

X_train_ss = ss.transform(X_train)

X_test_ss = ss.transform(X_test)



mms.fit(X_train)

X_train_mms = mms.transform(X_train)

X_test_mms = mms.transform(X_test)
from sklearn.svm import SVC



svc = SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,

   decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

   max_iter=-1, probability=False, random_state=3, shrinking=True,

   tol=0.001, verbose=False)

svc.fit(X_train, y_train)



submission_df['Survived'] = svc.predict(X_test)

submission_df.to_csv('submission_SVC_rbf_NonScaling.csv', index=False)
svc = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,

   decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

   max_iter=-1, probability=False, random_state=3, shrinking=True,

   tol=0.001, verbose=False)

svc.fit(X_train_ss, y_train)



submission_df['Survived'] = svc.predict(X_test_ss)

submission_df.to_csv('submission_SVC_rbf_StandardScaler.csv', index=False)
svc = SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,

   decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

   max_iter=-1, probability=False, random_state=3, shrinking=True,

   tol=0.001, verbose=False)

svc.fit(X_train_mms, y_train)



submission_df['Survived'] = svc.predict(X_test_mms)

submission_df.to_csv('submission_SVC_rbf_MinMaxScaler.csv', index=False)
svc = SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,

   decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',

   max_iter=-1, probability=False, random_state=None, shrinking=True,

   tol=0.001, verbose=False)

svc.fit(X_train, y_train)



submission_df['Survived'] = svc.predict(X_test)

submission_df.to_csv('submission_SVC_linear_NonScaling.csv', index=False)
svc = SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,

   decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',

   max_iter=-1, probability=False, random_state=None, shrinking=True,

   tol=0.001, verbose=False)

svc.fit(X_train_ss, y_train)



submission_df['Survived'] = svc.predict(X_test_ss)

submission_df.to_csv('submission_SVC_linear_StandardScaler.csv', index=False)
svc = SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,

   decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',

   max_iter=-1, probability=False, random_state=None, shrinking=True,

   tol=0.001, verbose=False)

svc.fit(X_train_mms, y_train)



submission_df['Survived'] = svc.predict(X_test_mms)

submission_df.to_csv('submission_SVC_linear_MinMaxScaler.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier



knc = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

            metric_params=None, n_jobs=1, n_neighbors=15, p=2,

            weights='uniform')

knc.fit(X_train, y_train)



submission_df['Survived'] = knc.predict(X_test)

submission_df.to_csv('submission_KNeighborsClassifier_NonScaling.csv', index=False)
knc = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

            metric_params=None, n_jobs=1, n_neighbors=12, p=2,

            weights='uniform')

knc.fit(X_train_ss, y_train)



submission_df['Survived'] = knc.predict(X_test_ss)

submission_df.to_csv('submission_KNeighborsClassifier_StandardScaler.csv', index=False)
knc = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

            metric_params=None, n_jobs=1, n_neighbors=8, p=2,

            weights='uniform')

knc.fit(X_train_mms, y_train)



submission_df['Survived'] = knc.predict(X_test_mms)

submission_df.to_csv('submission_KNeighborsClassifier_MinMaxScaler.csv', index=False)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

           penalty='l2', random_state=3, solver='liblinear', tol=0.0001,

           verbose=0, warm_start=False)

lr.fit(X_train, y_train)



submission_df['Survived'] = lr.predict(X_test)

submission_df.to_csv('submission_LogisticRegression_NonScaling.csv', index=False)
lr = LogisticRegression(C=10.0, class_weight=None, dual=False, fit_intercept=True,

           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

           penalty='l2', random_state=3, solver='liblinear', tol=0.0001,

           verbose=0, warm_start=False)

lr.fit(X_train_ss, y_train)



submission_df['Survived'] = lr.predict(X_test_ss)

submission_df.to_csv('submission_LogisticRegression_StandardScaler.csv', index=False)
lr = LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,

           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

           penalty='l2', random_state=3, solver='liblinear', tol=0.0001,

           verbose=0, warm_start=False)

lr.fit(X_train_mms, y_train)



submission_df['Survived'] = lr.predict(X_test_mms)

submission_df.to_csv('submission_LogisticRegression_MinMaxScaler.csv', index=False)
from sklearn.linear_model import Perceptron



ppn = Perceptron(alpha=1e-10, class_weight=None, eta0=1.0, fit_intercept=True,

       max_iter=10000, n_jobs=1, penalty=None, random_state=3,

       shuffle=True, tol=None, verbose=0, warm_start=False)

ppn.fit(X_train, y_train)



submission_df['Survived'] = ppn.predict(X_test)

submission_df.to_csv('submission_Perceptron_NonScaling.csv', index=False)
ppn = Perceptron(alpha=1e-10, class_weight=None, eta0=1.0, fit_intercept=True,

       max_iter=10000, n_jobs=1, penalty=None, random_state=3,

       shuffle=True, tol=None, verbose=0, warm_start=False)

ppn.fit(X_train_ss, y_train)



submission_df['Survived'] = ppn.predict(X_test_ss)

submission_df.to_csv('submission_Perceptron_StandardScaler.csv', index=False)
ppn = Perceptron(alpha=1e-10, class_weight=None, eta0=1.0, fit_intercept=True,

       max_iter=10000, n_jobs=1, penalty=None, random_state=3,

       shuffle=True, tol=None, verbose=0, warm_start=False)

ppn.fit(X_train_mms, y_train)



submission_df['Survived'] = ppn.predict(X_test_mms)

submission_df.to_csv('submission_Perceptron_MinMaxScaler.csv', index=False)
from sklearn.neural_network import MLPClassifier



mlpc = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,

        beta_2=0.999, early_stopping=False, epsilon=1e-08,

        hidden_layer_sizes=(30, 20, 10), learning_rate='constant',

        learning_rate_init=0.001, max_iter=200, momentum=0.9,

        nesterovs_momentum=True, power_t=0.5, random_state=3, shuffle=True,

        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,

        warm_start=False)

mlpc.fit(X_train, y_train)



submission_df['Survived'] = mlpc.predict(X_test)

submission_df.to_csv('submission_MLPClassifier_NonScaling.csv', index=False)
mlpc = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,

        beta_2=0.999, early_stopping=False, epsilon=1e-08,

        hidden_layer_sizes=(100,), learning_rate='constant',

        learning_rate_init=0.001, max_iter=200, momentum=0.9,

        nesterovs_momentum=True, power_t=0.5, random_state=3, shuffle=True,

        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,

        warm_start=False)

mlpc.fit(X_train_ss, y_train)



submission_df['Survived'] = mlpc.predict(X_test_ss)

submission_df.to_csv('submission_MLPClassifier_StandardScaler.csv', index=False)
mlpc = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,

        beta_2=0.999, early_stopping=False, epsilon=1e-08,

        hidden_layer_sizes=(50, 50), learning_rate='constant',

        learning_rate_init=0.001, max_iter=200, momentum=0.9,

        nesterovs_momentum=True, power_t=0.5, random_state=3, shuffle=True,

        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,

        warm_start=False)

mlpc.fit(X_train_mms, y_train)



submission_df['Survived'] = mlpc.predict(X_test_mms)

submission_df.to_csv('submission_MLPClassifier_MinMaxScaler.csv', index=False)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

             max_depth=7, max_features='auto', max_leaf_nodes=None,

             min_impurity_decrease=0.0, min_impurity_split=None,

             min_samples_leaf=1, min_samples_split=2,

             min_weight_fraction_leaf=0.0, n_estimators=14, n_jobs=1,

             oob_score=False, random_state=3, verbose=0, warm_start=False)

rfc.fit(X_train, y_train)



submission_df['Survived'] = rfc.predict(X_test)

submission_df.to_csv('submission_RandomForestClassifier_NonScaling.csv', index=False)
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

             max_depth=7, max_features='auto', max_leaf_nodes=None,

             min_impurity_decrease=0.0, min_impurity_split=None,

             min_samples_leaf=1, min_samples_split=2,

             min_weight_fraction_leaf=0.0, n_estimators=14, n_jobs=1,

             oob_score=False, random_state=3, verbose=0, warm_start=False)

rfc.fit(X_train_ss, y_train)



submission_df['Survived'] = rfc.predict(X_test_ss)

submission_df.to_csv('submission_RandomForestClassifier_StandardScaler.csv', index=False)
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

             max_depth=7, max_features='auto', max_leaf_nodes=None,

             min_impurity_decrease=0.0, min_impurity_split=None,

             min_samples_leaf=1, min_samples_split=2,

             min_weight_fraction_leaf=0.0, n_estimators=14, n_jobs=1,

             oob_score=False, random_state=3, verbose=0, warm_start=False)

rfc.fit(X_train_mms, y_train)



submission_df['Survived'] = rfc.predict(X_test_mms)

submission_df.to_csv('submission_RandomForestClassifier_MinMaxScaler.csv', index=False)