import pandas as pd
import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline


print("modules for data visualization imported")
train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
print("train and test dataset loaded")
print("features in train dataset \n")
train_dataset.info()
print("features in test dataset \n")
test_dataset.info()
train_dataset.head()
train_dataset.tail()
female_color = "#FA0000"

plt.subplot2grid((3,4), (0,3))
train_dataset.Sex[train_dataset.Survived == 1].value_counts(normalize=True).plot(kind="pie")
plt.subplot2grid((2, 3), (1, 0), colspan=2)
for x in [1, 2, 3]:
    train_dataset.Age[train_dataset.Pclass == x].plot(kind="kde")
    
plt.title("age distribution among different classes")
g = sb.FacetGrid(train_dataset, col='Survived')
g.map(plt.hist, 'Age', bins=20)
g.map(plt.plot, "Age", "Survived", marker=".", color="red")
train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


plt.subplot2grid((3,4), (1,0), colspan=4)
for x in [1, 2, 3]:
    train_dataset.Survived[train_dataset.Pclass == x].plot(kind="kde")
plt.legend()
plt.subplot2grid((3,4), (2,2))
train_dataset.Survived[(train_dataset.Sex == "female") & (train_dataset.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color="pink")
plt.title("Rich Women Survived")
plt.subplot2grid((3,4), (2,2))
train_dataset.Survived[(train_dataset.Sex == "male") & (train_dataset.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color="blue")
plt.title("Poor Men Survived")
train = train_dataset
train["Hypothesis"] = 0
train.loc[train.Sex == "female", "Hypothesis"] = 1

train["Result"] = 0
train.loc[train.Survived == train["Hypothesis"], "Result"] = 1

print(train["Result"].value_counts(normalize=True))
def data_process(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    
    
    
    data = data.drop(['Fare'], axis=1)
    data = data.drop(['Ticket'], axis=1)
    data = data.drop(['Cabin'], axis=1)
    freq_port = train_dataset.Embarked.dropna().mode()[0]
#     for dataset in train_dataset:
#         dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
#     for dataset in train_dataset:
#         dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


    data['Embarked'] = data['Embarked'].fillna(freq_port)

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    
    data = data.drop(['Name'], axis=1)
    
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
      
    data.loc[ data['Age'] <= 16, 'Age'] = int(0)
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age']
    
    return data
    
    
    
    
    
# train_dataset = data_process(train_dataset)
# train_dataset.head()
    
    
import utils


train_dataset = data_process(train_dataset)
train_dataset.head()



from sklearn import linear_model, preprocessing

target = train_dataset["Survived"].values
features = train_dataset[["Pclass", "Sex", "Age", "SibSp", "Parch"]].values

classfier = linear_model.LogisticRegression()
classifier_ = classfier.fit(features, target)

print(classifier_.score(features, target))
poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

classfier = linear_model.LogisticRegression()
classifier_ = classfier.fit(poly_features, target)

print(classifier_.score(poly_features, target))
from sklearn import tree

decision_tree = tree.DecisionTreeClassifier(random_state = 1)
decision_tree_ = decision_tree.fit(features, target)

print(decision_tree_.score(features, target))
from sklearn.ensemble import *
random_forest = RandomForestClassifier(n_estimators=100)

Y_train_dataset = train_dataset["Survived"]
X_train_dataset = train_dataset.drop("Survived", axis=1)
X_train_dataset = X_train_dataset.drop("PassengerId", axis=1)
X_train_dataset = X_train_dataset.drop("Hypothesis", axis=1)
X_train_dataset = X_train_dataset.drop("Result", axis=1)


X_train_dataset.head()
random_forest.fit(X_train_dataset, Y_train_dataset)


X_test_dataset = data_process(test_dataset)
X_test_dataset = X_test_dataset.drop("PassengerId", axis=1)
X_test_dataset.head()
predicted_value = random_forest.predict(X_test_dataset)

test_dataset_copy = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "PassengerId": test_dataset_copy["PassengerId"],
        "Survived": predicted_value
})

submission.to_csv('submission.csv', index=False)