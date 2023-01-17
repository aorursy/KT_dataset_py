import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

data_train.head()
def simplify_ages(df):

    guess_ages = np.zeros((2,3))

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = df[(df['Sex'] == i) & \

                                  (df['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

        for i in range(0, 2):

            for j in range(0, 3):

                df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1),\

                        'Age'] = guess_ages[i,j]

        df['Age'] = df['Age'].astype(int)

        

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_name(df):

    df['NamePrefix'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['NamePrefix'] = df['NamePrefix'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['NamePrefix'] = df['NamePrefix'].replace('Mlle', 'Miss')

    df['NamePrefix'] = df['NamePrefix'].replace('Ms', 'Miss')

    df['NamePrefix'] = df['NamePrefix'].replace('Mme', 'Mrs')

    return df    



def simplify_sex(df):

    df['Sex']=pd.get_dummies(df.Sex).drop('female',axis=1)

    return df



def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



def transform_features(df):

    df = simplify_sex(df)

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = drop_features(df)

    return df



data_train = transform_features(data_train)

data_test = transform_features(data_test)

data_train.head()
train_age = pd.get_dummies(data_train.Age).drop('Adult',axis=1)

test_age = pd.get_dummies(data_test.Age).drop('Adult',axis=1)

train_fare = pd.get_dummies(data_train.Fare).drop('1_quartile',axis=1)

test_fare = pd.get_dummies(data_test.Fare).drop('1_quartile',axis=1)

train_pclass = pd.get_dummies(data_train.Pclass)

test_pclass = pd.get_dummies(data_test.Pclass)

train_cabin = pd.get_dummies(data_train.Cabin).drop('N',axis=1)

test_cabin = pd.get_dummies(data_test.Cabin).drop('N',axis=1)

test_cabin['T'] = 0

train_name =pd.get_dummies(data_train.NamePrefix).drop('Mr',axis=1)

test_name =pd.get_dummies(data_test.NamePrefix).drop('Mr',axis=1)
train_X = pd.concat([data_train['Sex'],train_age,train_fare,train_pclass,train_cabin,train_name,data_train['SibSp'],data_train['Parch']],axis=1)

X_test = pd.concat([data_test['Sex'],test_age,test_fare,test_pclass,test_cabin,test_name,data_test['SibSp'],data_test['Parch']],axis=1)
train_y = data_train["Survived"]
def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        Xtrain, Xtest = train_X.values[train_index], train_X.values[test_index]

        ytrain, ytest = train_y.values[train_index], train_y.values[test_index]

        clf.fit(Xtrain, ytrain)

        predictions = clf.predict(Xtest)

        accuracy = accuracy_score(ytest, predictions)

        outcomes.append(accuracy)

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 
random_forest = RandomForestClassifier()

run_kfold(random_forest)
logreg = LogisticRegression()

run_kfold(logreg)
svc = SVC()

run_kfold(svc)
knn = KNeighborsClassifier(n_neighbors = 8)

run_kfold(knn)
linear_svc = LinearSVC()

run_kfold(linear_svc)
ids = data_test['PassengerId']

predictions = logreg.predict(X_test)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

#output.to_csv('submissions.csv', index = False)