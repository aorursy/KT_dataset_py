import numpy as np

import pandas as pd



# Plots.

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Preprocessing.

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import cross_val_score, GridSearchCV



# ML ALGORITHMS.

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.svm import SVC, SVR

import xgboost as xgb



# Validation.

from sklearn.metrics import mean_absolute_error, accuracy_score



import warnings

warnings.filterwarnings('ignore')
!ls '../input/titanic'
!head '../input/titanic/train.csv'
X_full_path = '../input/titanic/train.csv'

X_full = pd.read_csv(X_full_path, index_col='PassengerId')



X_full.Survived.isnull().sum()
X_full.head()
X_full.describe()
data = X_full.copy()



surv_dict = {0: 'No', 1: 'Yes'}

data['Survived'] = data['Survived'].map(surv_dict)
f_mask = data['Sex'] == 'female'

m_mask = data['Sex'] == 'male'

c1_mask = data['Pclass'] == 1

c2_mask = data['Pclass'] == 2

c3_mask = data['Pclass'] == 3



total = data['Survived'].value_counts().rename('Total')

female = data.loc[f_mask, 'Survived'].value_counts().rename('Female')

male = data.loc[m_mask, 'Survived'].value_counts().rename('Male')

class1 = data.loc[c1_mask, 'Survived'].value_counts().rename('Class 1')

class2 = data.loc[c2_mask, 'Survived'].value_counts().rename('Class 2')

class3 = data.loc[c3_mask, 'Survived'].value_counts().rename('Class 3')
rows1 = pd.concat([total, female, male, class1, class2, class3], sort=True, axis=1).T

rows1.T.plot.pie(subplots=True, figsize=(20, 5), autopct='%1.0f%%', legend=False)

plt.show()
female1 = data.loc[f_mask & c1_mask, 'Survived'].value_counts().rename('Female class 1')

female2 = data.loc[f_mask & c2_mask, 'Survived'].value_counts().rename('Female class 2')

female3 = data.loc[f_mask & c3_mask, 'Survived'].value_counts().rename('Female class 3')

male1 = data.loc[m_mask & c1_mask, 'Survived'].value_counts().rename('Male class 1')

male2 = data.loc[m_mask & c2_mask, 'Survived'].value_counts().rename('Male class 2')

male3 = data.loc[m_mask & c3_mask, 'Survived'].value_counts().rename('Male class 3')



rows2 = pd.concat([female1, female2, female3, male1, male2, male3], sort=True, axis=1).T

rows2.T.plot.pie(subplots=True, figsize=(20, 5), autopct='%1.0f%%', legend=False)

plt.show()
y = X_full.Survived

X = X_full.drop('Survived', axis=1)

X.head()
X.describe()
y.head()
X_test_path = '../input/titanic/test.csv'

X_test = pd.read_csv(X_test_path, index_col='PassengerId')

X_test.head()
X_test.describe()
def check_null_values(X, X_test):

    

    common_columns = set(X.columns).intersection(set(X_test.columns))

    null_sum = 0

    

    for column in common_columns:

        

        train_null = X[column].isnull()

        train_sum = train_null.sum()

        

        test_null = X_test[column].isnull()

        test_sum = test_null.sum()

        

        if train_sum + test_sum > 0:

            print(f'Null values in {column}:')

            if train_sum > 0:

                    print(f'--> train: {train_sum}')

            if test_sum > 0:

                    print(f'--> test: {test_sum}')

        

        null_sum += train_sum + test_sum

        

    if null_sum == 0:

        print('No null values!')

                

def generate_columns_for_null_values(X, X_test):

    

    common_columns = set(X.columns).intersection(set(X_test.columns))

    

    for column in common_columns:

        

        train_null = X[column].isnull()

        train_sum = train_null.sum()

        

        test_null = X_test[column].isnull()

        test_sum = test_null.sum()

        

        new_column = column + '_is_missing'

        

        if (train_sum > 0 and new_column not in X.columns and test_sum > 0 and new_column not in X_test.columns):

            X[new_column] = train_null

            print('New column generated for X: {}'.format(new_column))

            X_test[new_column] = test_null

            print('New column generated for X_test: {}'.format(new_column))
check_null_values(X, X_test)

generate_columns_for_null_values(X, X_test)
# NAME TITLE.

# Get anything (except spaces) between a space and a dot.

my_regex = r' ([^ ]*)[.]'



X['NameTitle'] = X['Name'].str.extract(my_regex)

X_test['NameTitle'] = X_test['Name'].str.extract(my_regex)



# CABIN LETTER.

X['CabinLetter'] = X['Cabin'].str[0]

X_test['CabinLetter'] = X_test['Cabin'].str[0]



# TICKET CODE.

# Get anything (including spaces) between a space and end of string.

my_regex = r'(.*) [^ ]*$'



X['TicketCode'] = X['Ticket'].str.extract(my_regex)

X_test['TicketCode'] = X_test['Ticket'].str.extract(my_regex)



# TICKET NUMBER.

# Get numbers (at least one) (except spaces) before end of string.

my_regex = r'(\d+[^ ])$'



X['TicketNumber'] = X['Ticket'].str.extract(my_regex)

X_test['TicketNumber'] = X_test['Ticket'].str.extract(my_regex)



X['TicketNumber'] = pd.to_numeric(X['TicketNumber'])

X_test['TicketNumber'] = pd.to_numeric(X_test['TicketNumber'])
check_null_values(X, X_test)
X['Embarked'].value_counts().plot(kind='bar')

plt.show()
# Only 2 missing values. Assign the mode.

mask = X['Embarked'].isnull()

X.loc[mask, 'Embarked'] = 'S'
# Everybody has to have a cabin... Too difficult right now, just empty it.

mask = X['Cabin'].isnull()

X.loc[mask, 'Cabin'] = ''

mask = X['CabinLetter'].isnull()

X.loc[mask, 'CabinLetter'] = ''

mask = X_test['Cabin'].isnull()

X_test.loc[mask, 'Cabin'] = ''

mask = X_test['CabinLetter'].isnull()

X_test.loc[mask, 'CabinLetter'] = ''
# Only 1 missing value. Assign the mean.

mask = X_test['Fare'].isnull()

X_test.loc[mask, 'Fare'] = X_test['Fare'].mean()
# If nan, no ticket code. Empty it.

mask = X['TicketCode'].isnull()

X.loc[mask, 'TicketCode'] = ''

mask = X_test['TicketCode'].isnull()

X_test.loc[mask, 'TicketCode'] = ''



# Same with ticket number

mask = X['TicketNumber'].isnull()

X.loc[mask, 'TicketNumber'] = ''

mask = X_test['TicketNumber'].isnull()

X_test.loc[mask, 'TicketNumber'] = ''
check_null_values(X, X_test)
# Some dependance: Pclass, SibSp, Parch, NameTitle

# No dependance: Sex, Embarked

# Danger: CabinLetter. Most of them are unknown.



cat_cols = ['Pclass', 'SibSp', 'Parch', 'NameTitle']



for col in cat_cols:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    X[['Age', col]].boxplot(by=col, ax=ax1)

    X_test[['Age', col]].boxplot(by=col, ax=ax2)

    plt.show()
# No dependance at all.



num_cols = ['Fare', 'TicketNumber']



for col in num_cols:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    X[['Age', col]].plot(x=col, y='Age', style='o', ax=ax1)

    X_test[['Age', col]].plot(x=col, y='Age', style='o', ax=ax2)

    plt.show()
# AGE: RF with features Pclass, SibSp, Parch, NameTitle.



age_columns = ['Pclass', 'SibSp', 'Parch', 'NameTitle']



mask = X['Age'].isnull()

age_X = X.loc[~mask, age_columns]

age_y = X.loc[~mask, 'Age']



mask_test = X_test['Age'].isnull()

age_X_test = X_test.loc[~mask_test, age_columns]

age_y_test = X_test.loc[~mask_test, 'Age']



# All passengers with age informed will be used in the cross-validation.

age_X = pd.concat([age_X, age_X_test], axis=0)

age_y = pd.concat([age_y, age_y_test], axis=0)



# All passengers with no age informed are stored in these datasets.

age_missing_X = X.loc[mask, age_columns]

age_missing_X_test = X_test.loc[mask_test, age_columns]

age_missing_X = pd.concat([age_missing_X, age_missing_X_test], axis=0)
def prepro(X, X_test, cols):

    

    # 1.

    # ONE HOT ENCODING.

    oh_columns = cols

    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



    # Categorical columns.

    oh_X = pd.DataFrame(oh_encoder.fit_transform(X[oh_columns]))

    oh_X.index = X.index

    oh_X_test = pd.DataFrame(oh_encoder.transform(X_test[oh_columns]))

    oh_X_test.index = X_test.index



    # Numerical columns.

    num_X = X.drop(oh_columns, axis=1)

    num_X_test = X_test.drop(oh_columns, axis=1)



    # Join.

    X = pd.concat([num_X, oh_X], axis=1)

    X_test = pd.concat([num_X_test, oh_X_test], axis=1)



    # 2.

    # SCALING.

    sc = StandardScaler()

    sc_X = pd.DataFrame(sc.fit_transform(X))

    sc_X_test = pd.DataFrame(sc.fit_transform(X_test))



    sc_X.index = X.index

    sc_X_test.index = X_test.index



    return sc_X, sc_X_test
oh_columns = ['NameTitle']

age_X, age_missing_X = prepro(age_X, age_missing_X, oh_columns)
models = [

    ('DT', DecisionTreeRegressor(random_state=0)),

    ('RF', RandomForestRegressor(random_state=0)),

    ('SVM', SVR()),

    ('XGB', xgb.XGBRegressor(random_state=0, verbosity=0))

]
hm = pd.DataFrame()



for name, model in models:

    scores = cross_val_score(model, age_X, age_y, cv=5, scoring='neg_mean_absolute_error')

    i = 0

    for score in scores:

        hm = hm.append({'row': int(i), 'model': name, 'score': -score}, ignore_index=True)

        i = i + 1



hm = hm.pivot(index='row', columns='model', values='score')
plt.figure(figsize=(10, 5))

sns.boxplot(data=hm)

plt.show()
def grid_search(my_model, my_parameters, X, y):

    

    # 1. Grid Search.

    model = GridSearchCV(my_model, my_parameters, cv=5, scoring='neg_mean_absolute_error')

    model.fit(X, y)



    # 2. Results.

    best = model.best_params_.copy()

    print('Best parameters: {}'.format(best))

    

    # 3. Heatmap.

    params = model.cv_results_['params'].copy()

    means = model.cv_results_['mean_test_score']

    keys = list(my_parameters[0].keys())

    

    if len(my_parameters) == 1 and len(keys) == 2:

        

        hm = pd.DataFrame()

        for param, mean in zip(params, means):

            param['MAE'] = -mean

            hm = hm.append(param, ignore_index=True)

        hm = hm.pivot(index=keys[0], columns=keys[1], values='MAE')

        

        plt.figure(figsize=(15, 10))

        sns.heatmap(hm, cmap='Spectral', annot=True, fmt='.3f')

        plt.show()

    

    return best
my_model = DecisionTreeRegressor()

my_parameters = [

    {'max_features': np.linspace(0.1, 1.0, 10),

    'max_depth': [5*(1 + i) for i in range(5)]}

]



DT_params = grid_search(my_model, my_parameters, age_X, age_y)
my_model = RandomForestRegressor()

my_parameters = [

    {'n_estimators': [5*(1 + i) for i in range(10)],

    'max_depth': [5*(1 + i) for i in range(6)]}

]



RF_params = grid_search(my_model, my_parameters, age_X, age_y)
my_model = SVR(gamma='scale', epsilon=0.01)

my_parameters = [

    {'kernel': ['poly', 'rbf'],

    'C': [5*(5 + i) for i in range(10)]}

]



SVM_params = grid_search(my_model, my_parameters, age_X, age_y)
my_model = xgb.XGBRegressor(verbosity=0)

my_parameters = [

    {'n_estimators': [50, 100, 1000, 2000],

    'learning_rate': [0.01, 0.05, 0.1, 0.2]}

]



XGB_params = grid_search(my_model, my_parameters, age_X, age_y)
models = [

    ('DT', DecisionTreeRegressor(**DT_params)),

    ('RF', RandomForestRegressor(**RF_params)),

    ('SVM', SVR(**SVM_params)),

    ('XGB', xgb.XGBRegressor(**XGB_params, verbosity=0))

]
hm = pd.DataFrame()



for name, model in models:

    scores = cross_val_score(model, age_X, age_y, cv=5, scoring='neg_mean_absolute_error')

    i = 0

    for score in scores:

        hm = hm.append({'row': int(i), 'model': name, 'score': -score}, ignore_index=True)

        i = i + 1



hm = hm.pivot(index='row', columns='model', values='score')
hm.mean()
plt.figure(figsize=(10, 5))

sns.boxplot(data=hm)

plt.show()
model = xgb.XGBRegressor(**XGB_params, verbosity=0)

model.fit(age_X, age_y)



pred = pd.DataFrame({'Age': model.predict(age_missing_X)})

pred.index = age_missing_X.index
X.loc[mask, 'Age'] = X.apply(lambda row: pred.loc[row.index, 'Age'])

X_test.loc[mask_test, 'Age'] = X_test.apply(lambda row: pred.loc[row.index, 'Age'])
check_null_values(X, X_test)
X.head()
cat_cols = [col for col in X.columns if X[col].dtype=='object']
print('CATEGORICAL COLUMNS CARDINALITY under 20:\n')

for col in cat_cols:

    all_cat = set(X[col]).union(set(X_test[col]))

    cardinality = len(all_cat)

    if cardinality < 20:

        print('{}: {}'.format(col, cardinality))

        print('--> {}'.format(all_cat))

        

print('\n\n')

print('CATEGORICAL COLUMNS CARDINALITY over 20:\n')

for col in cat_cols:

    all_cat = set(X[col]).union(set(X_test[col]))

    cardinality = len(all_cat)

    if not cardinality < 20:

        print('{}: {}'.format(col, cardinality))
drop_cols = ['Name', 'Ticket', 'Cabin', 'TicketCode', 'TicketNumber']

X = X.drop(drop_cols, axis=1)

X_test = X_test.drop(drop_cols, axis=1)
# SEX: is a categorical column of cardinality 2, so it can be label encoded.

sex_dict = {'female': 0, 'male': 1}

X['Sex'] = X['Sex'].map(sex_dict)

X_test['Sex'] = X_test['Sex'].map(sex_dict)
# *_is_missing columns can be relabeled.

bool_cols = [col for col in X.columns if X[col].dtype=='bool']

bool_dict = {False: 0, True: 1}



for col in bool_cols:

    X[col] = X[col].map(bool_dict)

    X_test[col] = X_test[col].map(bool_dict)
oh_columns = ['Embarked', 'NameTitle', 'CabinLetter']

X, X_test = prepro(X, X_test, oh_columns)
models = [

    ('DT', DecisionTreeClassifier(random_state=0)),

    ('RF', RandomForestClassifier(random_state=0)),

    ('SVM', SVC()),

    ('XGB', xgb.XGBClassifier(random_state=0, verbosity=0))

]
hm = pd.DataFrame()



for name, model in models:

    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')

    i = 0

    for score in scores:

        hm = hm.append({'row': int(i), 'model': name, 'score': -score}, ignore_index=True)

        i = i + 1



hm = hm.pivot(index='row', columns='model', values='score')
plt.figure(figsize=(10, 5))

sns.boxplot(data=hm)

plt.show()
my_model = DecisionTreeClassifier()

my_parameters = [

    {'max_features': np.linspace(0.1, 1.0, 10),

    'max_depth': [5*(1 + i) for i in range(5)]}

]



DT_params = grid_search(my_model, my_parameters, X, y)
my_model = RandomForestClassifier()

my_parameters = [

    {'n_estimators': [5*(1 + i) for i in range(10)],

    'max_depth': [5*(1 + i) for i in range(6)]}

]



RF_params = grid_search(my_model, my_parameters, X, y)
my_model = SVC(gamma='scale')

my_parameters = [

    {'kernel': ['linear', 'poly', 'rbf'],

    'C': [0.001, 0.01, 0.04, 0.05, 0.06, 0.1, 1]}

]



SVM_params = grid_search(my_model, my_parameters, X, y)
my_model = xgb.XGBClassifier(verbosity=0)

my_parameters = [

    {'n_estimators': [50, 100, 1000, 2000],

    'learning_rate': [0.01, 0.05, 0.1, 0.2]}

]



XGB_params = grid_search(my_model, my_parameters, X, y)
models = [

    ('DT', DecisionTreeClassifier(**DT_params)),

    ('RF', RandomForestClassifier(**RF_params)),

    ('SVM', SVC(**SVM_params)),

    ('XGB', xgb.XGBClassifier(**XGB_params, verbosity=0))

]
hm = pd.DataFrame()



for name, model in models:

    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')

    i = 0

    for score in scores:

        hm = hm.append({'row': int(i), 'model': name, 'score': -score}, ignore_index=True)

        i = i + 1



hm = hm.pivot(index='row', columns='model', values='score')
hm.mean()
plt.figure(figsize=(10, 5))

sns.boxplot(data=hm)

plt.show()
clf = xgb.XGBClassifier(**XGB_params, verbosity=0)

clf.fit(X, y)
!head '../input/titanic/gender_submission.csv'
submission = pd.DataFrame({'Survived': clf.predict(X_test)})

submission.index = X_test.index

submission['Survived'] = submission['Survived']

submission.to_csv('submission.csv')
!head '../working/submission.csv'