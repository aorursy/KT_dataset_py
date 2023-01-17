import os

import pandas as pd

from statistics import mode

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC, SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from numpy import argmax

from matplotlib import pyplot as plt

import numpy as np
input_dir = '../input/titanic/'

train_fn = os.path.join(input_dir, 'train.csv')

test_fn = os.path.join(input_dir, 'test.csv')



output_dir = '../input/output/'



train_drop_cols = {'Cabin', 'SibSp', 'Parch', 'Ticket', 'Name'}
def one_hot_encode(input_df, col):

    le = LabelEncoder()

    ohe = OneHotEncoder()

    label_encoded = le.fit_transform(input_df[col])

    label_encoded = label_encoded.reshape(len(label_encoded), 1)

    onehot_encoded = ohe.fit_transform(label_encoded).toarray()

    ohe_col_names = ['_'.join([col, _class]) for _class in le.classes_]

    df = pd.DataFrame(onehot_encoded, columns=ohe_col_names)

    input_df.drop(col, axis=1, inplace=True)

    for col in ohe_col_names:

        input_df[col] = df[col]

    return input_df
train = pd.read_csv(train_fn)

test = pd.read_csv(test_fn)
data = pd.concat([train, test], ignore_index=True)
nan_cols = set(data.columns[((data.isnull().sum() / data.shape[0])*100) > 0]) - train_drop_cols

print("NaN Columns: ", nan_cols)



median_by_sex = data[['Sex', 'Age']].groupby('Sex').median()['Age'].to_dict()

mode_by_pclass = 'S'

median_by_pclass_and_emb = 8.0500

impute_dict = {

    'Age': median_by_sex,

    'Embarked': mode_by_pclass,

    'Fare': median_by_pclass_and_emb

}

print("Imputation Dictionery: \n", impute_dict)



for col, val in impute_dict.items():

    if col == 'Age':

        for sex, val2 in impute_dict[col].items():

            data.loc[data['Sex'] == sex, col] = data.loc[data['Sex'] == sex, col].fillna(val2)

    else:

        data[col] = data[col].fillna(val)
data['FamilySize'] = data[['SibSp', 'Parch']].sum(axis=1)
cat_cols = {'Sex', 'Embarked'}

for col in cat_cols:

    print(col)

    data = one_hot_encode(data, col)
data.drop(train_drop_cols, axis=1, inplace=True)
new_train = data[data['Survived'].notnull()]

new_test = data[data['Survived'].isnull()].drop('Survived', axis=1)
X = new_train.drop(['PassengerId', 'Survived'], axis=1)

y = new_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
model_dict = {

    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=9),

    'LogisticRegression': LogisticRegression(max_iter=100000, random_state=0),

    'LinearSVC': LinearSVC(max_iter=100000, random_state=0),

    'SVC': SVC(gamma=0.1, C=0.3),

    'GaussianNB': GaussianNB(),

    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=4, random_state=0),

    'RandomForestClassifier': RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0),

    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=50, max_depth=1, random_state=0),

    'MLPClassifier': MLPClassifier(random_state=0, solver='lbfgs', hidden_layer_sizes=[10, 10], alpha=0.0100),

    'XGBClassifier': XGBClassifier(random_state=0, learning_rate=0.01, n_estimators=50, subsample=0.8, max_depth=4)

}
data_list = list()

for name, model in model_dict.items():

    data_dict = dict()

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)

    test_score = model.score(X_test, y_test)

    data_dict['model'] = name

    data_dict['train_score'] = train_score

    data_dict['test_score'] = test_score

    data_list.append(data_dict)

score_df = pd.DataFrame(data_list)

score_df['score_diff'] = score_df['train_score'] - score_df['test_score']

model_df = score_df.sort_values(['test_score'], ascending=[False])

model_df
m_name = 'XGBClassifier'

output_file = 'Submission_{}.csv'.format(m_name)

output_path = os.path.join('../input/output/', output_file)

model = model_dict[m_name].fit(X, y)

predictions = model.predict(new_test.drop('PassengerId', axis=1))

test['Survived'] = [int(i) for i in predictions.tolist()]

test[['PassengerId', 'Survived']].to_csv(output_file, index=False)