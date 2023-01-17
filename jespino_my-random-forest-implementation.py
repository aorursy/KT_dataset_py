import pandas as pd

import numpy as np

%matplotlib inline

from matplotlib.pyplot import plot, scatter

from sklearn.learning_curve import learning_curve

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import normalize
def guess_data(old_data):

    data = old_data.copy()

    data.loc[data.Embarked.isnull(), "Embarked"] = data.Embarked.dropna().mode().values

    data.loc[data.Age.isnull(), "Age"] = data.Age.dropna().mean()

    data.loc[data.Fare.notnull() & data.Cabin.notnull(), "Fare"] = data.loc[data.Fare.notnull() & data.Cabin.notnull(), "Fare"] / data.loc[data.Fare.notnull() & data.Cabin.notnull(), "Cabin"].map(lambda x: len(x.split()))

    data.loc[data.Fare.isnull(), "Fare"] = data.Fare.dropna().median()

    return data

    

def data_clean_up(old_data):

    data = old_data.copy()

    data['Sex'] = data.Sex.map({"female":0, "male":1}).astype(int)

    data['Child'] = data.Age < 16

    data['Young'] = data.Age.between(16,30)

    data['Adult'] = data.Age.between(31,59)

    data['HasFamily'] = data.SibSp > 0

    data['HasParChild'] = data.Parch > 0

    data['CabinA'] = data.Cabin.map(lambda x: x[0] == "A" if not x else False)  

    data['CabinB'] = data.Cabin.map(lambda x: x[0] == "B" if not x else False)    

    data['CabinC'] = data.Cabin.map(lambda x: x[0] == "C" if not x else False) 

    data['CabinD'] = data.Cabin.map(lambda x: x[0] == "D" if not x else False)

    data['CabinE'] = data.Cabin.map(lambda x: x[0] == "E" if not x else False)

    data['CabinF'] = data.Cabin.map(lambda x: x[0] == "F" if not x else False)

    data['CabinG'] = data.Cabin.map(lambda x: x[0] == "G" if not x else False)

    data['EmbarkedC'] = data.Embarked == "C"

    data['EmbarkedQ'] = data.Embarked == "Q"

    return data.drop(['Cabin', "Ticket", "Name", "PassengerId", "Age", "Embarked", "SibSp", "Parch"], axis=1)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train
data = data_clean_up(guess_data(train))

test_data = data_clean_up(guess_data(test))
forest = RandomForestClassifier(n_estimators=100, random_state=42)

y = data.Survived

X = data.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

forest.fit(normalize(X_train),y_train)

forest.score(normalize(X_test), y_test)
curve = learning_curve(forest, normalize(X), y, train_sizes=np.linspace(0.1, 1.0, 10))

plot(curve[0],curve[1], "r-", curve[0], curve[2], "b-")
pred = forest.predict(normalize(test_data))

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": pred

    })



submission.to_csv('titanic.csv', index=False)