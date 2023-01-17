from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB



import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
train = pd.read_csv('../input/train.csv')
train.info()
train['Age'] = train['Age'].fillna(round(train['Age'].mean()))
train = train.drop(['Cabin'], axis=1)
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'female' else 0)
def set_embarqued_order(city):

    if city == 'Q':

        return 1

    

    if city == 'S':

        return 2



    return 3



train['Embarked'] = train['Embarked'].apply(lambda x: set_embarqued_order(x))
train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

train = train.dropna()
train['parents'] = train['SibSp'] + train['Parch']
plt.figure(figsize=[15, 7])

sns.heatmap(train.corr(), annot=True, fmt='.2f', cmap=sns.diverging_palette(999, 999, n=20), vmax=1, vmin=-1)

plt.title('Correlação entre as variaveis')

plt.show()
qty_suvived = len(train.query('Survived == 1')['Survived'])

qty_not_suvived = len(train.query('Survived == 0')['Survived'])

total = qty_suvived + qty_not_suvived
plt.title('Distribuição de sobreviventes e não sobrevientes')

plt.pie(

    [round(qty_suvived/total, 2), round(qty_not_suvived/total, 2)],

    labels=['Sobreviventes', 'Não Sobreviventes'],

    autopct='%.2f')

plt.show()
columns = train.drop(['Survived'], axis=1).columns

X = train[columns]

y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
def validate_model(y_test, y_predict):

    accuracy = accuracy_score(y_test, y_predict)

    precision = precision_score(y_test, y_predict)

    recall = recall_score(y_test, y_predict)

    

    plt.figure(figsize=[20, 5])

    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='.2f')

    plt.show()



    print('Acuracia: %.2f' % accuracy)

    print('Precisão: %.2f' % precision)

    print('Recall: %.2f' % recall)
naive_bayes = GaussianNB()

naive_bayes = naive_bayes.fit(X_train, y_train)



validate_model(y_test, naive_bayes.predict(X_test))
logistic_regression = LogisticRegression(solver='liblinear')

logistic_regression = logistic_regression.fit(X_train, y_train)



validate_model(y_test, logistic_regression.predict(X_test))
random_forest = RandomForestClassifier(n_estimators=70, random_state=0)

random_forest = random_forest.fit(X_train, y_train)



validate_model(y_test, random_forest.predict(X_test))
validation = pd.read_csv('../input/test.csv')
validation['Sex'] = validation['Sex'].apply(lambda x: 1 if x == 'female' else 0)

validation['Embarked'] = validation['Embarked'].apply(lambda x: set_embarqued_order(x))

validation['parents'] = validation['SibSp'] + validation['Parch']
validation.info()
validation['Age'] = validation['Age'].fillna(validation['Age'].mean())

validation['Fare'] = validation['Fare'].fillna(validation['Fare'].mean())
pd.DataFrame({

    'PassengerId': validation['PassengerId'],

    'Survived': random_forest.predict(validation[columns])

}).to_csv('./validation.csv', index=False)