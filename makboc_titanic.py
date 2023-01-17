# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import csv as csv

import seaborn as sns



from sklearn import linear_model as lm

from sklearn import metrics, ensemble

from sklearn.cross_validation import train_test_split

from sklearn import decomposition

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.feature_selection import RFE



from itertools import combinations

pd.options.display.max_columns = 2000
def update_columns(df, test=False):

    if test:

        df.columns = [

            'passanger_id', 'p_class', 'name', 'sex', 'age', 'sib_sp',

            'parch', 'ticket', 'fare', 'cabin', 'embarked'

        ]

    else:        

        df.columns = [

            'passanger_id', 'survived', 'p_class', 'name', 'sex', 'age', 'sib_sp',

            'parch', 'ticket', 'fare', 'cabin', 'embarked'

        ]

    

def read_train_data():

    df = pd.read_csv('../input/train.csv')

    update_columns(df)

    return df



def read_test_data():

    df = pd.read_csv('../input/test.csv')

    update_columns(df, test=True)

    return df
def enrich_df(df):

    df = pd.concat([df, pd.get_dummies(df.p_class, prefix='class')], axis=1)

    df['age_2'] = df.age.apply(lambda x: x ** 2).fillna(0)

    df['age'] = df.age.fillna(0)

    df['has_age'] = (df.age > 0).apply(int)

    df = pd.concat([df, pd.get_dummies(df.sex, prefix='sex')], axis=1)

    df['siblings_1'] = (df.sib_sp == 1).apply(int)

    df['siblings_2'] = (df.sib_sp == 2).apply(int)

    df['siblings_3'] = (df.sib_sp == 3).apply(int)

    df['siblings_4p'] = (df.sib_sp >= 4).apply(int)

    df['parch_1'] = (df.parch == 1).apply(int)

    df['parch_2'] = (df.parch == 2).apply(int)

    df['parch_3'] = (df.parch == 3).apply(int)

    df['parch_4p'] = (df.parch >= 4).apply(int)

    df['fare'] = df.fare.fillna(0)

    df['log_fare'] = df.fare.apply(lambda x: np.log1p(x))

    df['num_cabins'] = df.cabin.apply(lambda x: 0 if x is np.nan else len(x.split(' ')))

    df['cabin_group'] = df.cabin.apply(lambda x: 'u' if x is np.nan else x[0].lower())

    df = pd.concat([df, pd.get_dummies(df.cabin_group, prefix='cabin_group')], axis=1)

    df['embarked'] = df.embarked.fillna('U').apply(lambda x: x.lower())

    df = pd.concat([df, pd.get_dummies(df.embarked, prefix='embarked')], axis=1)

    return df
train = read_train_data()

test = read_test_data()



train = enrich_df(train)

test = enrich_df(test)
train.describe()
features = [

    'class_1', 'class_2', 'has_age', 'age',

    #'age_2', 

    'sex_male',

    'sib_sp',

    #'siblings_1', 'siblings_2', 'siblings_3', 'siblings_4p',

    'parch', 

    #'parch_1', 'parch_2', 'parch_3', 'parch_4p',

    'log_fare',

    'num_cabins',

    'cabin_group_b', 'cabin_group_c', 'cabin_group_d', 'cabin_group_e',

    'cabin_group_f', 'cabin_group_a',# 'cabin_group_u',

    'embarked_q', 'embarked_s'

]

y = train.survived.values

X = train[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_valid = test[features]



X_train = X_train.copy()

X_test = X_test.copy()

y_train = y_train.copy()

y_test = y_test.copy()

X_valid = X_valid.copy()
X_train.head()
X_train.age.apply(lambda x: np.sqrt(x)).plot(kind='hist');
X_train.age_2.apply(lambda x: np.power(x, 0.3)).plot(kind='hist');
X_train['age'] = X_train.age.apply(lambda x: np.sqrt(x))

X_test['age'] = X_test.age.apply(lambda x: np.sqrt(x))

X_valid['age'] = X_valid.age.apply(lambda x: np.sqrt(x))



#X_train['age_2'] = X_train.age_2.apply(lambda x: np.power(x, 0.3))

#X_test['age_2'] = X_test.age_2.apply(lambda x: np.power(x, 0.3))

#X_valid['age_2'] = X_valid.age_2.apply(lambda x: np.power(x, 0.3))
age_mean = np.mean(X_train['age'][X_train['age'].gt(0)])

age_std = np.std(X_train['age'][X_train['age'].gt(0)])



#age_2_mean = np.mean(X_train['age_2'][X_train['age_2'].gt(0)])

#age_2_std = np.std(X_train['age_2'][X_train['age_2'].gt(0)])



log_fare_mean = np.mean(X_train['log_fare'])

log_fare_std = np.std(X_train['log_fare'])



print([age_mean, age_std])

#print([age_2_mean, age_2_std])

print([log_fare_mean, log_fare_std])
X_train['age'] = X_train.age.apply(lambda x: (x - age_mean) / age_std if x > 0 else x)

X_test['age'] = X_test.age.apply(lambda x: (x - age_mean) / age_std if x > 0 else x)

X_valid['age'] = X_valid.age.apply(lambda x: (x - age_mean) / age_std if x > 0 else x)



#X_train['age_2'] = X_train.age_2.apply(lambda x: (x - age_2_mean) / age_2_std if x > 0 else x)

#X_test['age_2'] = X_test.age_2.apply(lambda x: (x - age_2_mean) / age_2_std if x > 0 else x)

#X_valid['age_2'] = X_valid.age_2.apply(lambda x: (x - age_2_mean) / age_2_std if x > 0 else x)



X_train['log_fare'] = X_train.log_fare.apply(lambda x: (x - log_fare_mean) / log_fare_std)

X_test['log_fare'] = X_test.log_fare.apply(lambda x: (x - log_fare_mean) / log_fare_std)

X_valid['log_fare'] = X_valid.log_fare.apply(lambda x: (x - log_fare_mean) / log_fare_std)
for var_a, var_b in combinations(X_train.columns, 2):

    X_train[var_a + '_' + var_b] = X_train[var_a] * X_train[var_b]

    X_test[var_a + '_' + var_b] = X_test[var_a] * X_test[var_b]

    X_valid[var_a + '_' + var_b] = X_valid[var_a] * X_valid[var_b]
X_train.describe()
lr_model = lm.LogisticRegression()

lr_model = lr_model.fit(X=X_train, y=y_train)



y_train_hat_lr = lr_model.predict(X_train)

y_test_hat_lr = lr_model.predict(X_test)



print(metrics.classification_report(y_train, y_train_hat_lr))

print(metrics.classification_report(y_test, y_test_hat_lr))
gb_model = ensemble.GradientBoostingClassifier()

gb_model = gb_model.fit(X=X_train, y=y_train)

y_train_hat_gb = gb_model.predict(X_train)

y_test_hat_gb = gb_model.predict(X_test)

y_valid_hat_gb = gb_model.predict(X_valid)

print(metrics.classification_report(y_train, y_train_hat_gb))

print(metrics.classification_report(y_test, y_test_hat_gb))
rf_model = ensemble.RandomForestClassifier()

rf_model = rf_model.fit(X=X_train, y=y_train)

y_train_hat_rf = rf_model.predict(X_train)

y_test_hat_rf = rf_model.predict(X_test)

print(metrics.classification_report(y_train, y_train_hat_rf))

print(metrics.classification_report(y_test, y_test_hat_rf))
ab_model = ensemble.AdaBoostClassifier()

ab_model = ab_model.fit(X=X_train, y=y_train)

y_train_hat_ab = ab_model.predict(X_train)

y_test_hat_ab = ab_model.predict(X_test)

print(metrics.classification_report(y_train, y_train_hat_ab))

print(metrics.classification_report(y_test, y_test_hat_ab))
bc_model = ensemble.BaggingClassifier()

bc_model = bc_model.fit(X=X_train, y=y_train)

y_train_hat_bc = bc_model.predict(X_train)

y_test_hat_bc = bc_model.predict(X_test)

print(metrics.classification_report(y_train, y_train_hat_bc))

print(metrics.classification_report(y_test, y_test_hat_bc))
et_model = ensemble.ExtraTreesClassifier()

et_model = et_model.fit(X=X_train, y=y_train)

y_train_hat_et = et_model.predict(X_train)

y_test_hat_et = et_model.predict(X_test)

print(metrics.classification_report(y_train, y_train_hat_et))

print(metrics.classification_report(y_test, y_test_hat_et))
ml_model = MLPClassifier(activation='relu', hidden_layer_sizes=(10),

                      batch_size=200, learning_rate='adaptive', max_iter=1000)

ml_model = ml_model.fit(X=X_train, y=y_train)

y_train_hat_ml = ml_model.predict(X_train)

y_test_hat_ml = ml_model.predict(X_test)

print(metrics.classification_report(y_train, y_train_hat_ml))

print(metrics.classification_report(y_test, y_test_hat_ml))
sv_model = SVC(probability=True, C=1, gamma=0.1, kernel='rbf')

sv_model = sv_model.fit(X=X_train, y=y_train)

y_train_hat_sv = sv_model.predict(X_train)

y_test_hat_sv = sv_model.predict(X_test)

print(metrics.classification_report(y_train, y_train_hat_sv))

print(metrics.classification_report(y_test, y_test_hat_sv))
n_folds = 5

cv = StratifiedKFold(n_folds)

C_range = 10.0 ** np.arange(-4, 4)

gamma_range = 10.0 ** np.arange(-4, 4)

param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist())

svr = SVC(kernel='rbf')

grid = GridSearchCV(estimator=svr, param_grid=param_grid, 

                    n_jobs=1, cv=list(cv.split(X_train, y_train)))

grid.fit(X_train, y_train)

print("The best classifier is: ", grid.best_estimator_)

#print(grid.grid_scores_)
vc_model = ensemble.VotingClassifier(estimators=[

    ('lr_model', lr_model),

    ('gb_model', gb_model),

    ('rf_model', rf_model),

    ('ab_model', ab_model),

    ('bc_model', bc_model),

    ('et_model', et_model),

    ('ml_model', ml_model),

    ('sv_model', sv_model)

], voting='soft')

vc_model = vc_model.fit(X=X_train, y=y_train)

print(metrics.classification_report(y_train, vc_model.predict(X_train)))

print(metrics.classification_report(y_test, vc_model.predict(X_test)))
X_temp = X_train.copy()

X_temp['y'] = y_train

X_temp['y_hat'] = y_train_hat_lr
X_temp.columns
model = vc_model



passenger_id = test.passanger_id.values

survived = model.predict(X_valid)



predictions_file = open("my_submission.csv", "w")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(["PassengerId","Survived"])

open_file_object.writerows(zip(passenger_id, survived))

predictions_file.close()



print(check_output(["ls"]).decode("utf8"))