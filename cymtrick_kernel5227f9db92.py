import numpy as np

import pandas as pandas

import re as re

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder



train = pandas.read_csv('../input/titanic/train.csv', header = 0, dtype={'Age': np.float64})

test  = pandas.read_csv('../input/titanic/test.csv' , header = 0, dtype={'Age': np.float64})





print(train.info())   
#  Fare and Survival mean cut to four categories

train['Fare'] = train['Fare'].fillna(train['Fare'].median())

train['Fare'] = pandas.qcut(train['Fare'], 4)

print(train[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean())
# Embarked and Survival mean

train['Embarked'] = train['Embarked'].fillna('S')

print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
# filling the empty ages and making mean with the survival rate

average_age = train['Age'].mean()

std_age = train['Age'].std()

null_count_age = train['Age'].isnull().sum()



null_random_list = np.random.randint(average_age - std_age, average_age + std_age , size = null_count_age)



train['Age'] = pandas.cut(train['Age'],5)



print (train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean())

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
train['alone'] = 0

train.loc[train['FamilySize'] == 1, 'alone'] = 1

print (train[['alone', 'Survived']].groupby(['alone'], as_index=False).mean())
sns.heatmap(train.isnull())
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


def clean_data(data):

 

    # Transforming the Age into {0,4}

    average_age = data['Age'].mean()

    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    std_age = data['Age'].std()

    null_count_age = data['Age'].isnull().sum()

    null_random_list = np.random.randint(average_age - std_age, average_age + std_age , size = null_count_age)

    data['Age'][np.isnan(data['Age'])] = null_random_list

    data['Age'] = data['Age'].astype(int)

    data.loc[ data['Age'] <= 16, 'Age'] = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[ data['Age'] > 64, 'Age'] = 4

    

    

    # Adding the alone column as feature {0,1}

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    data['alone'] = 0

    data.loc[data['FamilySize'] == 1, 'alone'] = 1

    

    # Transforming the Embarked into {0,2}

    data['Embarked'] = data['Embarked'].fillna('S')

    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    data['Sex'] = data['Sex'].map({"female":0,"male":1})

    

    # Transforming the fare 10 {0,5}

#     (-0.001, 7.91]  0.197309

#     (7.91, 14.454]  0.303571

#     (14.454, 31.0]  0.454955

#     (31.0, 512.329]  0.581081

    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454),'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31.0),'Fare']   = 2

    data.loc[(data['Fare'] > 31.0) & (data['Fare'] <= 512.329),'Fare'] = 3

    data.loc[data['Fare'] >= 512.329, 'Fare'] = 5

    data['Fare'] = data['Fare'].astype(int)

    

    # Dropping the unnecessary columns

    drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']

    data = data.drop(drop_columns, axis = 1)

    return data



train_data = pandas.read_csv('../input/titanic/train.csv', header = 0, dtype={'Age': np.float64})

test_data = pandas.read_csv('../input/titanic/test.csv', header = 0, dtype={'Age': np.float64})

train = clean_data(train_data)

test = clean_data(test_data)

print(train)
#Classifier learning



from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB



X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test

print(X_train.info())

print(X_test.info())

classifiers = [LogisticRegression(),RandomForestClassifier(),Perceptron(),

               SGDClassifier(),DecisionTreeClassifier(),KNeighborsClassifier(),SVC(),LinearSVC(),GaussianNB(),GradientBoostingClassifier(),LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis()]
class_strings = ['LogisticRegression','RandomForestClassifier','Perceptron',

               'SGDClassifier','DecisionTreeClassifier','NeighborsClassifier','SVC','LinearSVC',

                 'GaussianNB','GradientBoostingClassifier','LinearDiscriminantAnalysis','QuadraticDiscriminantAnalysis']



col_scores = ["Classifier", "Accuracy"]

scores_pd = pandas.DataFrame(columns=col_scores)

scores = []

for x in classifiers:

    name = x.__class__.__name__

    model = x

    model.fit(X_train,Y_train)

    predictions = model.predict(X_test)

    score = model.score(X_train, Y_train) * 100

    scores.append(score)





for z in range(12):

        data = pandas.DataFrame([[class_strings[z],int(scores[z])]], columns=col_scores)

        scores_pd = scores_pd.append(data)

print(scores_pd)

sns.barplot(x='Accuracy', y='Classifier', data=scores_pd, color='g')
final_classifier1 =  RandomForestClassifier(n_estimators=100, oob_score = True)

final_classifier1.fit(X_train, Y_train)

result = final_classifier1.predict(X_test)
remove_features = pandas.DataFrame({'feature':X_train.columns,'unneccessary_features':np.round(final_classifier1.feature_importances_,3)})

remove_features = remove_features.sort_values('unneccessary_features',ascending=True).set_index('feature')

remove_features.head(100)
X_train = train.drop("Survived", axis=1)

X_train = X_train.drop("alone",axis=1)

Y_train = train["Survived"]

X_test  = test.drop("alone",axis=1)
final_classifier1 =  RandomForestClassifier(n_estimators=100, oob_score = True)

final_classifier1.fit(X_train, Y_train)

result = final_classifier1.predict(X_test)

score = final_classifier1.score(X_train, Y_train) * 100

print(score)
final_classifier2 =  DecisionTreeClassifier()

final_classifier2.fit(X_train, Y_train)

result = final_classifier2.predict(X_test)

remove_features = pandas.DataFrame({'feature':X_train.columns,'unneccessary_features':np.round(final_classifier2.feature_importances_,3)})

remove_features = remove_features.sort_values('unneccessary_features',ascending=True).set_index('feature')

remove_features.head(100)
X_train = train.drop("Survived", axis=1)

X_train = X_train.drop("alone",axis=1)

Y_train = train["Survived"]

X_test  = test.drop("alone",axis=1)



final_classifier2 =  DecisionTreeClassifier()

final_classifier2.fit(X_train, Y_train)

score = final_classifier2.score(X_train, Y_train) * 100

print(score)