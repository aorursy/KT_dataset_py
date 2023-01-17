import numpy as np
import pandas as pd
# read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
full_data = [train, test]
train.head(3)
test.head(3)
# find null values
for data in full_data:
    print(data.isnull().sum(), '\n')
# Correlation between the 'survived' column and 
# the other numerical columns in train data
train.corr()['Survived'].sort_values(ascending=False)
for data in full_data:
    
    # Age
    mean = data['Age'].mean()
    std = data['Age'].std()
    data.loc[:, 'Age'] = data['Age'].apply(lambda x: np.random.randint((mean-std), (mean+std)) 
                                           if pd.isnull(x) else x)

    data['Age'] = data['Age'].astype(int)
    
    
    # Cabin
    data.drop(columns='Cabin', inplace=True)
    
    
    # Fare
    data.loc[data['Fare'].isnull(), 'Fare'] = data['Fare'].median()
    
    
    # Embarked
    data.loc[data['Embarked'].isnull(), 'Embarked'] = max(data.groupby('Embarked')['Embarked']
                                                          .agg('count').items())[0]
for data in full_data:
    print(data.isnull().sum(), '\n')
# use train data's distribution.
# create a list of pandas intervals to apply onto test data.
# add an interval in case there's person in test data older than in train data
train['CategoricalAge'] = pd.qcut(train['Age'], 10)
categorical_age = sorted(train.groupby('CategoricalAge').groups, key=lambda x: x.left, reverse=False)
categorical_age.append(pd.Interval(left=categorical_age[-1].right, right=np.inf))
categorical_age = list(enumerate(categorical_age))
train.drop(columns=['CategoricalAge'], inplace=True)
categorical_age
# do the same as the age
train['CategoricalFare'] = pd.qcut(train['Fare'], 10)
categorical_fare = sorted(train.groupby('CategoricalFare').groups, key=lambda x: x.left, reverse=False)
categorical_fare.append(pd.Interval(left=categorical_fare[-1].right, right=np.inf))
categorical_fare = list(enumerate(categorical_fare))
train.drop(columns=['CategoricalFare'], inplace=True)
categorical_fare
# Name
for data in full_data:
    
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    data.loc[data['Title'].isin(['Sir', 'Countess', 'Dona', 'Col', 'Dr', 'Jonkheer', 
                                 'Major', 'Don', 'Rev', 'Capt']), 'Title'] = 'Rare'
    data.loc[data['Title'].isin(['Lady', 'Miss', 'Mlle']), 'Title'] = 'Ms'
    data.loc[data['Title'].isin(['Mme']), 'Title'] = 'Mrs'
# Family size

for data in full_data:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
for data in full_data:
    
    # enumerate
    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    data['Title'] = data['Title'].map({"Mr": 1, "Ms": 2, "Mrs": 3, "Master": 4, "Rare": 5}).astype(int)
    data['Age'] = data['Age'].map(lambda x: [i for i, interval in categorical_age 
                                             if x in interval][0]).astype(int)
    data['Fare'] = data['Fare'].map(lambda x: [i for i, interval in categorical_fare
                                               if x in interval][0]).astype(int)
    
    
    # drop
    data.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize'], inplace=True)
train.head(3)
# import modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# Split data into train & test data
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
# Support Vector Machines classifier

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)
# Linear svm classifier

lsvc = LinearSVC()

lsvc.fit(X_train, Y_train)

Y_pred = lsvc.predict(X_test)

lsvc.score(X_train, Y_train)
# Random forrest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
# K-Neighbors classifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, Y_train)
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

gaussian.score(X_train, Y_train)
# Seeing them together

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]


acc_dict = {}
for clf in classifiers:
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    acc = clf.score(X_train, Y_train)
    acc_dict.update({clf.__class__.__name__:acc})

    
p = pd.DataFrame.from_dict(acc_dict, orient='index')
p.columns = ['acc']
p.sort_values(by='acc', ascending=False)
Y_pred = random_forest.predict(X_test)
test = pd.read_csv('../input/test.csv')

sub = submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
sub[:3]
