import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



def fill_na(data):

    data.Age = data.Age.fillna(data.Age.median()) 

    data.Embarked = data.Embarked.fillna('S')

    data.Fare = data.Fare.fillna(data.Fare.mean())

    

    return data



df_train = fill_na(df_train)

df_test = fill_na(df_test)



def convert_vector(data):

    data.Sex = data.Sex.replace(['male', 'female'], [0, 1])

    data.Embarked = data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])

    

    return data



df_train = convert_vector(df_train)

df_test = convert_vector(df_test)
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



models = []



models.append(('LogisticRegression', LogisticRegression(solver='liblinear')))

models.append(('RandomForest', RandomForestClassifier(n_estimators=10)))

models.append(('DecisionTree', DecisionTreeClassifier()))

models.append(('SVM', SVC(gamma='scale')))

models.append(('GaussianNB', GaussianNB()))



results = []

names = []

for name, model in models:

    result = cross_val_score(model, df_train[predictors], df_train['Survived'], cv=3)

    names.append(name)

    results.append(result)

for i in range(len(names)):

    print(names[i], results[i].mean())
forest_parameters = {

    'n_estimators': [1, 11, 50, 100, 500, 1000, 5000],

    'max_depth': [3, 5, 7, 11, 13, 15],

    'random_state': [0],

}

gsc_forest = GridSearchCV(RandomForestClassifier(), forest_parameters, cv=10, iid=False)

gsc_forest.fit(df_train[predictors], df_train['Survived'])

print(gsc_forest.best_params_)



X_train, X_test, y_train, y_test = train_test_split(df_train[predictors], df_train['Survived'], test_size=0.3, random_state=0)



rf = RandomForestClassifier(n_estimators=5000, max_depth=11, criterion='entropy', random_state=0)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

y_true = y_test

print('accuracy score : ' + str(accuracy_score(y_true, y_pred)))

prediction = rf.predict(df_test[predictors])
df_out = pd.read_csv('../input/test.csv', encoding='utf-8')

df_out['Survived'] = prediction

df_out[['PassengerId', 'Survived']].to_csv('submission_rf.csv', index=False)