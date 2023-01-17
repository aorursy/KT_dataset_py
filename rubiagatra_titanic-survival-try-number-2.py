import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

%matplotlib inline
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.head()
plt.style.use('fivethirtyeight')

sns.barplot(x='Sex', y='Survived', data=data_train)
sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data_train)
data_train['Age'].hist(bins=70)
data_train['Age'].quantile([0, 0.25, .75, .9])
def age_categorize(age):

    """Categorize Age by three Category (Young, Adult, Senior) using Age Quantile"""

    if age < 20.125:

        return 'Young'

    elif age >= 20.125 and age <= 38.000:

        return 'Adult'

    else:

        return 'Senior'        
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean())
data_train['Age_Category'] = data_train['Age'].apply(age_categorize)
data_train.info()
sns.barplot(x='Age_Category', y='Survived', data=data_train)
sns.factorplot(x='Age_Category', kind='count',hue='Survived' ,data=data_train)
data_train.head()
sns.distplot(data_train.Fare)
data_train.Fare.quantile([.25, .5, .75])
def fare_categorize(age):

    """Categorize Fare by three Category (Low, Middle, High) using Age Quantile"""

    if age < 7.9104:

        return 'Low'

    elif age >= 14.4542 and age <= 31.0000:

        return 'Middle'

    else:

        return 'High' 
data_train['Fare_Category'] = data_train.Fare.apply(fare_categorize)
data_train.head()
sns.barplot(x='Fare_Category', y='Survived', data=data_train)
sns.barplot(x='Embarked', y='Survived', data=data_train)
data_train['Is_Alone'] = data_train['SibSp'] + data_train['Parch'] == 0
sns.barplot(x='Is_Alone', y='Survived', data=data_train)
data_train.head()
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head(6)
def preproccesing_data(df):

    df['Age'] = df['Age'].fillna(df['Age'].mean())

    df['Age_Category'] = (df['Age'].apply(age_categorize)).map({'Young':0, 'Adult':1, 'Senior':2})

    df['Fare_Category'] = (df.Fare.apply(fare_categorize)).map({'Low':0, 'Middle':1, 'High':2})

    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

    df['Sex'] = df['Sex'].map({'male':0, 'female':1}).astype(int)

    df_final = df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1)

    return df_final
train_final = preproccesing_data(train_df)

test_final = preproccesing_data(test_df)
test_final
X_train = train_final.drop("Survived", axis=1).fillna(0.0)

Y_train = train_final["Survived"]

X_test  = test_final.copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)