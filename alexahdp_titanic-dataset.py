import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score

from sklearn.metrics import mean_squared_error, r2_score, recall_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron

from xgboost import XGBClassifier



import warnings

warnings.filterwarnings('ignore')
train_raw = pd.read_csv('../input/train.csv')

test_raw = pd.read_csv('../input/test.csv')
train_raw.head(5)
train_raw.info()
train = train_raw

test = test_raw



male_mean_age = train[train['Sex'] == 'male']['Age'].mean()

female_mean_age = train[train['Sex'] == 'female']['Age'].mean()



train['Age'].loc[(train['Sex'] == 'male') & train['Age'].isna()] = male_mean_age

train['Age'].loc[(train['Sex'] == 'female') & train['Age'].isna()] = female_mean_age



test['Age'].loc[(test['Sex'] == 'male') & test['Age'].isna()] = male_mean_age

test['Age'].loc[(test['Sex'] == 'female') & test['Age'].isna()] = female_mean_age



train['Sex'] = pd.factorize(train['Sex'])[0]

test['Sex'] = pd.factorize(test['Sex'])[0]
train = train.drop('Cabin', axis=1)

train = train.drop('Ticket', axis=1)

train = train.drop('PassengerId', axis=1)
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

test['Embarked'] = test['Embarked'].fillna(train['Embarked'].mode()[0])



train['Fare'] = train['Fare'].fillna(train['Fare'].mode()[0])

test['Fare'] = test['Fare'].fillna(train['Fare'].mode()[0])



train['Embarked'] = pd.factorize(train['Embarked'])[0]

test['Embarked'] = pd.factorize(test['Embarked'])[0]
train['Age'] = train['Age'].astype(int)

test['Age'] = test['Age'].astype(int)
# === После ряда обучений ====

# Пришла пора заняться feature engineering-ом

# изучив поле Name, можно увидеть, что все пассажиры имеют приставку Mr, Miss, ...

# Логично предположить, что джентельмены уступали места дамам в спасательных шлюпках

# Проверим это



train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
# Уменьшим вариативность поля Title



v_dict = train['Title'].value_counts()



for key in v_dict.keys():

    if (v_dict[key] < 10):

        train['Title'] = train['Title'].replace(key, 'Smth')

        test['Title'] = test['Title'].replace(key, 'Smth')



train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')



pd.crosstab(train['Title'], train['Sex'])
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
train['Title'] = pd.factorize(train['Title'])[0]

test['Title'] = pd.factorize(test['Title'])[0]



train = train.drop('Name', axis=1)
train.info()
corr_matrix = train.corr()

sns.heatmap(corr_matrix, annot=True, fmt=".1f", linewidths=.5);
cols = ['Pclass', 'Sex', 'Fare', 'Title']

# cols = train.columns[train.columns != 'Survived']

cols
x_train, x_valid, y_train, y_valid = train_test_split(

    train[cols],

    train['Survived'],

    test_size=0.01, # 0.2

    random_state=42

)
# scaling данных не дал никаких результатов

scaler = StandardScaler()



x_train = scaler.fit_transform(x_train)

x_valid = scaler.transform(x_valid)
def evaluate_model(model_name, model):

    model.fit(x_train, y_train)

    y_predicted = model.predict(x_valid)



    acc = accuracy_score(y_valid, y_predicted)

    prec = precision_score(y_valid, y_predicted)

    rec = recall_score(y_valid, y_predicted)



    print(model_name, ': {acc:', round(acc, 2), 'prec: ', round(prec, 2), 'rec: ', round(rec, 2), '}')
models = {

    'LogisticRegression': LogisticRegression(),

    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=0, max_depth=5),

    'RandomForestClassifier': RandomForestClassifier(random_state=0, n_estimators=100, max_depth=4),

    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),

    'SVC': SVC(kernel='linear', gamma='scale'),

    'Perceptron': Perceptron(tol=1e-3, random_state=0),

    'xgboost': XGBClassifier(n_estimators=1000, learning_rate=0.05)

}



for model_name in models:

    evaluate_model(model_name, models[model_name])
vc = VotingClassifier(estimators = [

    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=0, max_depth=5)),

    ('RandomForestClassifier', RandomForestClassifier(random_state=0, n_estimators=100, max_depth=4)),

    ('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=5)),

    ('xgboost', XGBClassifier(n_estimators=1000, learning_rate=0.05))],

    voting='soft'

)



evaluate_model('VotingClassifier', vc)
test_scaled = scaler.transform(test[cols])

result = vc.predict(test_scaled)



result_dt = pd.DataFrame({

    'PassengerId': test_raw['PassengerId'],

    'Survived': result

})

result_dt.to_csv('submission.csv', index=False)