#importing libraries files

import numpy as np

import pandas as pd

import re

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC, SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
PATH = '../input/titanic/'



#show the list of files inside titanic_data directorory

!ls {PATH}
# loading training data

training_data = pd.read_csv(f"{PATH}/train.csv") 



#loading test data

test_data = pd.read_csv(f"{PATH}/test.csv")
training_data.info()

#training_data.columns

training_data.head(1)
#finding the missing values in data

total = training_data.isnull().sum().sort_values(ascending = False)

percent_1 = training_data.isnull().sum() / training_data.isnull().count() * 100     # sum add the values in each cell of a column while count returns the number of rows after neglecting only None value in each cell

percent_2 = round(percent_1,1).sort_values(ascending=False)

missing_data = pd.concat([total,percent_2],axis=1,keys=['Total','%'])

missing_data

# finding probability for each

pd.crosstab(training_data.Survived, training_data.Sex, margins= True, margins_name = 'Total', normalize= True)
#### Here 11 columns are input features i.e  PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

#### Column Survived is the output features

## * Processing Sex Feature

# sex have only two values i.e Male and Female. So we can convert it into boolean value

sex = {"male":0, "female":1}

data = [training_data, test_data]

for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(sex)

# Now Sex feature has been converted into numeric values

training_data.head(2)
# Name feature is approximately unique  So we can drop it. But before we can exract title from Name.

for dataset in data:

    dataset['Title'] = dataset['Name'].str.extract('(\w+)\.', expand = False)

    

    # replacing many titles with Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

pd.crosstab(training_data['Title'], training_data['Sex'], margins = True, margins_name = 'Total')
# titles tells about the age group and also relates with survival

training_data[['Title', 'Survived']].groupby(['Title'],as_index = False ).mean()
# converting tiles into int value



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

# removing Name features    

training_data = training_data.drop(['Name'],axis=1)

test_data = test_data.drop(['Name'],axis=1)

training_data.head(2)
data = [training_data,test_data]



for dataset in data:

    mean_age = training_data['Age'].mean()

    std_age =  test_data['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    #generate random age between mean age and std age

    random_age = np.random.randint(mean_age - std_age, mean_age + std_age, size = age_null_count)

    

    #replacing nan value

    dataset['Age'][np.isnan(dataset['Age'])] = random_age

    dataset['Age'] = dataset['Age'].astype(int)



training_data.isnull().sum()
training_data['Embarked'].describe()

training_data.groupby(by="Embarked").count()
common_val = 'S'

embark = {'S':0,'C':1,'Q':2}

data = [training_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_val)

    

    #replacing text values with int

    dataset['Embarked'] = dataset['Embarked'].map(embark)

    

#checking null value in Embarked features

training_data['Embarked'].isnull().sum()
training_data.head(5)
training_data['Ticket'].describe()
# Since Ticket features mostly contains unique values so we can neglect it.

training_data = training_data.drop(['Ticket'],axis= 1)

test_data = test_data.drop(['Ticket'], axis = 1)
training_data.head(5)
data = [training_data, test_data]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
training_data.head(5)
training_data.groupby(by = 'Cabin').count()
deck = {'A':1, 'B' :2, 'C':4, 'D':5,'E':6,'F':7,'G':8,'T':9}

data = [training_data, test_data]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("T0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

    

# we can now drop the cabin feature

training_data = training_data.drop(['Cabin'], axis=1)

test_data = test_data.drop(['Cabin'], axis=1)
#making the input features and output feautures training data

X_train = training_data.drop(['PassengerId','Survived'],axis= 1)

y_train = training_data.Survived



X_test = test_data.drop(['PassengerId'], axis = 1)

#making the input features  test data

X_train.head(1)

#y_train
# training using logistic regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)



# predicting result 

y_pred_log = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
#training using Linear SVC

lin_svc = LinearSVC()

lin_svc.fit(X_train, y_train)



y_pred_svc = lin_svc.predict(X_test)

acc_lin_svc = round(lin_svc.score(X_train, y_train) * 100, 2)

acc_lin_svc
#training using SVM

svm = SVC()

svm.fit(X_train, y_train)



y_pred_svm = svm.predict(X_test)

acc_svm = round(svm.score(X_train, y_train) * 100, 2)

acc_svm
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred_tree = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred_forest = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
# arranging model based on their score

modelScore = pd.DataFrame({

    'Model':['LogisticRegression','LinearSVC','SVC','DecisionTreeClassifier','RandomForestClassifier'],

    'Score':[acc_log,acc_lin_svc,acc_svm,acc_decision_tree,acc_random_forest]

})

modelScore.sort_values(by='Score', ascending=False)
#making submission file

submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": Y_pred_forest

    })

submission.to_csv('submission.csv', index=False)

submission