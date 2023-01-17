import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.head())
print(train.shape, test.shape)
print(train.columns)
print(test.columns)

print(train.dtypes)
print(test.dtypes)
print(train.describe())
survived_male = train.Survived[train.Sex == "male"].value_counts()
survived_female = train.Survived[train.Sex == "female"].value_counts()

print(survived_male)
print(survived_female)

dataFrame = pd.DataFrame({"male": survived_male, "female": survived_female})
dataFrame.plot(kind = "bar", stacked = True)
plt.title("Survived by Sex")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()
train["Age"].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution")
plt.show()

train[train.Survived == 0]["Age"].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who did not survived")
plt.show()

train[train.Survived == 1]["Age"].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who survived")
plt.show()
train["Fare"].hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution")
plt.show()

train[train.Survived == 0]["Fare"].hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution of people who did not survived")
plt.show()

train[train.Survived == 1]["Fare"].hist()
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title("Fare distribution of people who survived")
plt.show()
train["Pclass"].hist()
plt.show()
survived_p1 = train.Survived[train["Pclass"] == 1].value_counts()
survived_p2 = train.Survived[train["Pclass"] == 2].value_counts()
survived_p3 = train.Survived[train["Pclass"] == 3].value_counts()

dataFrame = pd.DataFrame({"p1": survived_p1, "p2": survived_p2, "p3": survived_p3})
print(dataFrame)
dataFrame.plot(kind = "bar", stacked = True)
plt.title("Survived by Pclass")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()
survived_S = train.Survived[train["Embarked"] == "S"].value_counts()
survived_C = train.Survived[train["Embarked"] == "C"].value_counts()
survived_Q = train.Survived[train["Embarked"] == "Q"].value_counts()

dataFrame = pd.DataFrame({"S": survived_S, "C": survived_C, "Q": survived_Q})
print(dataFrame)
dataFrame.plot(kind = "bar", stacked = True)
plt.title("Survived by Embarked")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()
origin_y_train = train["Survived"]
origin_X_train = train[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
origin_X_test = test[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
print(origin_X_train.shape)
print(origin_X_train)
def fill_NaN(data):
    data_copy = data.copy(deep = True)
    data_copy["Age"] = data_copy["Age"].fillna(data_copy["Age"].median())
    data_copy["Fare"] = data_copy["Fare"].fillna(data_copy["Fare"].median())
    data_copy["Pclass"] = data_copy["Pclass"].fillna(data_copy["Pclass"].median())
    data_copy["Sex"] = data_copy["Sex"].fillna("male")
    data_copy["Embarked"] = data_copy["Embarked"].fillna("S")
    return data_copy

# X_train['Embarked'].value_counts()

origin_X_train = fill_NaN(origin_X_train)
origin_X_test = fill_NaN(origin_X_test)
print(origin_X_train)
def transfer_sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy["Sex"] == "female", "Sex"] = 0
    data_copy.loc[data_copy["Sex"] == "male", "Sex"] = 1
    return data_copy

origin_X_train = transfer_sex(origin_X_train)
origin_X_test = transfer_sex(origin_X_test)
def transfer_embarked(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy["Embarked"] == "S", "Embarked"] = 0
    data_copy.loc[data_copy["Embarked"] == "C", "Embarked"] = 1
    data_copy.loc[data_copy["Embarked"] == "Q", "Embarked"] = 2
    return data_copy

origin_X_train = transfer_embarked(origin_X_train)
origin_X_test = transfer_embarked(origin_X_test)
from sklearn.model_selection import train_test_split
X_train, X_vali, y_train, y_vali = train_test_split(origin_X_train, origin_y_train, random_state = 0, test_size = 0.2)
print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_scores = []
k_range = range(1, 51)
for k in k_range:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_vali)
    score = accuracy_score(y_vali, predictions)
    k_scores.append(score)
plt.plot(k_range, k_scores)
plt.xlabel("k for KNN")
plt.ylabel("Accuracy on validation set")
plt.show()

# accuracy最大的那个k值
bestK = np.array(k_range)[np.array(k_scores).argsort()[-1]]
print("bestK =", bestK)
# 预测
clf = KNeighborsClassifier(n_neighbors = bestK)
clf.fit(origin_X_train, origin_y_train)
predictions = clf.predict(origin_X_test)
print(predictions)
df = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
df.to_csv("submission.csv", header = True)