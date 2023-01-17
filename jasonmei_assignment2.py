import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import os
print(os.listdir("../input"))
DATA_PATH = "../input"

train = pd.read_csv(DATA_PATH + "/train.csv")
test = pd.read_csv(DATA_PATH + "/test.csv")

train = train.reindex(np.random.permutation(train.index))
# Any results you write to the current directory are saved as output.
train.describe()
train.info()

test.info()
train.head(100)
sex_mapping = {'female': 0, 'male': 1}
train['Sex'] = train['Sex'].map(sex_mapping)
sns.countplot(train["Embarked"])
train['Embarked'] = train["Embarked"].fillna("S")
train['Embarked'] = train['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).fillna(3).astype(int)
print("age min:{}, max:{}".format(train['Age'].min(), train['Age'].max()))

def get_title(name):
    if "Mr." in name:
        return "Mr."
    elif "Mrs." in name:
        return "Mrs."
    elif "Miss." in name:
        return "Miss."
    elif "Master." in name:
        return "Master."
    else:
        return ""
train["Title"] = train["Name"].apply(lambda n : get_title(n))
train.groupby(['Title'])["Age"].mean()
train["Age"] = train.groupby("Title")["Age"].transform(lambda x: x.fillna(x.mean()))
train['Title'] = train['Title'].map({'Master.': 0, 'Miss.': 1, 'Mr.': 2,'Mrs.':3}).fillna(4).astype(int)
train.groupby("Pclass")["Fare"].mean()
train["Fare"] = train.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.mean()))
train.head()
corr = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corr, vmax=0.9, square=True, annot=True)
def preprocess_features(data):
    processed_features = data.copy()
    processed_features = processed_features.drop("PassengerId",axis=1)
    if 'Survived' in data.columns:
        processed_features = processed_features.drop("Survived",axis=1)
    processed_features = processed_features.drop("Name",axis=1)
    processed_features = processed_features.drop("Ticket",axis=1)
    processed_features = processed_features.drop("Cabin",axis=1)
    return processed_features
def preprocess_targets(data):
    output_targets = pd.DataFrame()
    output_targets["Survived"] = data["Survived"]
    return output_targets
x=preprocess_features(train)
y=preprocess_targets(train)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
model_linear = LogisticRegression()
model_linear.fit(x_train, y_train.values.ravel())
train_score = model_linear.score(x_train, y_train )
print("training accuracy: {}".format(train_score))
test_score = model_linear.score(x_test, y_test )
print("training accuracy: {}".format(test_score))

model_rf = RandomForestClassifier(random_state=1)
model_rf.fit(x_train, y_train.values.ravel())
train_score = model_rf.score(x_train, y_train )
print("training accuracy: {}".format(train_score))
test_score = model_rf.score(x_test, y_test )
print("training accuracy: {}".format(test_score))

def preprocess_data(_data):
    data = _data.copy()
    sex_mapping = {'female': 0, 'male': 1}
    data['Sex'] = data['Sex'].map(sex_mapping)
    data['Embarked'] = data["Embarked"].fillna("S")
    data['Embarked'] = data['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).fillna(3).astype(int)
    data["Title"] = data["Name"].apply(lambda n : get_title(n))
    data["Age"] = data.groupby("Title")["Age"].transform(lambda x: x.fillna(x.mean()))
    data['Title'] = data['Title'].map({'Master.': 0, 'Miss.': 1, 'Mr.': 2,'Mrs.':3}).fillna(4).astype(int)
    data["Fare"] = data.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.mean()))
    return data
TEST = preprocess_data(test)
X = preprocess_features(TEST)
def output_file(Y,filename):
    output = pd.DataFrame()
    output["PassengerId"] = test["PassengerId"].copy()
    output["Survived"] = Y
    output.to_csv(filename, index=False, header=["PassengerId", "Survived"])
Y = model_linear.predict(X)
output_file(Y, "submission_linear.csv")
Y = model_rf.predict(X)
output_file(Y, "submission_rf.csv")