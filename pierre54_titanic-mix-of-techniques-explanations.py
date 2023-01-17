import pandas as pd

train = pd.read_csv("../input/train.csv")

train.info()
train.describe()
train.head()
Pid = "PassengerId"

Class = "Survived"

Continuous = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

Text = ["Name", "Ticket", "Cabin"]

Discrete = "Embarked"

Dummy = "Sex"
train["Age"] = train["Age"].fillna(train["Age"].mean())

train["Embarked"] = train["Embarked"].fillna(train["Embarked"].value_counts().idxmax())



# ---- From now, I will use data in categorical variables ----- #

train_pid = train[Pid]

train_class = train[Class]

train_continuous = train[Continuous]

train_text = train[Text]

train_discrete = train[Discrete]

train_dummy = train[Dummy]
import re



def getTitle(x):

    result = re.compile(r'.*?,(.*?)\.').search(x)

    if result:

        return result.group(1).strip()

    else:

        return ''



train_family_size = train["SibSp"] + train["Parch"] + 1

train_continuous = pd.concat([train_continuous, train_family_size], axis=1)



train_title = train["Name"].apply(getTitle)

title_mapping = {"Mr": 0, "Miss": 0, "Mrs": 0, "Master": 0, "Dr": 1, "Rev": 1, "Major": 1, "Col": 1, "Mlle": 0, "Mme": 0, "Don": 1, "Lady": 1, "the Countess": 1, "Jonkheer": 1, "Sir": 1, "Capt": 1, "Ms": 0, "Dona": 1, "": 0}



for k,v in title_mapping.items():

    train_title[train_title == k] = v



train_continuous = pd.concat([train_continuous, train_title], axis=1)
from sklearn import preprocessing as pp

minmax_scaler = pp.MinMaxScaler((0,1)).fit(train_continuous)

train_continuous = pd.DataFrame(minmax_scaler.transform(train_continuous))
train_discrete = pd.get_dummies(train_discrete)



lb = pp.LabelBinarizer()

lb.fit(train_dummy)

train_dummy = pd.DataFrame(lb.transform(train_dummy))



# ---- For information ... ----- #

train_discrete.head()
X = pd.concat([train_continuous, train_discrete, train_dummy], axis=1)
import numpy as np

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

from sklearn import linear_model as lm



rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X, train_class)



knc = KNeighborsClassifier()

Cs = np.linspace(1, 19, 10).astype(int)

neigh = GridSearchCV(estimator=knc, param_grid=dict(n_neighbors=Cs), cv=10, n_jobs=-1)

neigh.fit(X, train_class)



mlp = MLPClassifier()

mlp.fit(X, train_class)



svc = svm.SVC()

Cs = np.logspace(-6, 2)

svc = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10, n_jobs=-1)

svc.fit(X, train_class)



lr = lm.LogisticRegression()

lr.fit(X, train_class)
test = pd.read_csv("../input/test.csv")



test["Age"] = test["Age"].fillna(train["Age"].mean())

test["Fare"] = test["Fare"].fillna(train["Fare"].mean())



test_pid = test[Pid]

test_continuous = test[Continuous]

test_text = test[Text]

test_discrete = test[Discrete]

test_dummy = test[Dummy]



test_family_size = test["SibSp"] + test["Parch"] + 1

test_continuous = pd.concat([test_continuous, test_family_size], axis=1)



test_title = test["Name"].apply(getTitle)



for k,v in title_mapping.items():

    test_title[test_title == k] = v



test_continuous = pd.concat([test_continuous, test_title], axis=1)



test_continuous = pd.DataFrame(minmax_scaler.transform(test_continuous))



test_discrete = pd.get_dummies(test_discrete)



test_dummy = pd.DataFrame(lb.transform(test_dummy))



X = pd.concat([test_pid, test_continuous, test_discrete, test_dummy], axis=1)
import csv as csv

result_file = open("result.csv", "w")

result_file_obj = csv.writer(result_file)

result_file_obj.writerow(["PassengerId", "Survived"])

for idx, row in X.iterrows():

	if(rfc.predict(row[1::].reshape(1, -1))[0] + neigh.predict(row[1::].reshape(1, -1))[0] + mlp.predict(row[1::].reshape(1, -1))[0] + svc.predict(row[1::].reshape(1, -1))[0] + lr.predict(row[1::].reshape(1, -1))[0] >= 3):

		result_file_obj.writerow([row["PassengerId"].astype(int), 1])

	else:

		result_file_obj.writerow([row["PassengerId"].astype(int), 0])



result_file.close()