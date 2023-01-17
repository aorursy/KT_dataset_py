import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/titanic/train.csv")

testset = pd.read_csv("../input/titanic/test.csv")



dataset.columns
dataset.head()
print(dataset.dtypes)
print(dataset.describe())
#Check Gender

Survived_m = dataset.loc[dataset.Sex == 'male', 'Survived'].value_counts()

Survived_f = dataset.loc[dataset.Sex == 'female', 'Survived'].value_counts()



df = pd.DataFrame({'male': Survived_m, 'female': Survived_f})

df.plot(kind = 'bar', stacked = True)

plt.title("Survived by sex")

plt.xlabel("Survived")

plt.ylabel("Count")

plt.show()
# Check Age

dataset['Age'].hist()

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Number")

plt.show()



dataset.loc[dataset.Survived == 0, 'Age'].hist()

plt.title("Age Distribution of people who did not survive")

plt.xlabel("Age")

plt.ylabel("Number")

plt.show()



dataset.loc[dataset.Survived == 1, 'Age'].hist()

plt.title("Age Distribution of people who survived")

plt.xlabel("Age")

plt.ylabel("Number")

plt.show()
# Check Fare

dataset['Fare'].hist()

plt.title("Fare Distribution")

plt.xlabel("Fare")

plt.ylabel("Number")

plt.show()



dataset.loc[dataset.Survived == 0, 'Fare'].hist()

plt.title("Fare Distribution of people who did not survive")

plt.xlabel("Fare")

plt.ylabel("Number")

plt.show()



dataset.loc[dataset.Survived == 1, 'Fare'].hist()

plt.title("Fare Distribution of people who survived")

plt.xlabel("Fare")

plt.ylabel("Number")

plt.show()
# Check Pclass

dataset['Pclass'].hist()

plt.show()

print(dataset['Pclass'].isnull().values.any())



Survived_p1 = dataset.loc[dataset.Pclass == 1, 'Survived'].value_counts()

Survived_p2 = dataset.loc[dataset.Pclass == 2, 'Survived'].value_counts()

Survived_p3 = dataset.loc[dataset.Pclass == 3, 'Survived'].value_counts()



df = pd.DataFrame({'p1': Survived_p1, "p2": Survived_p2, "p3": Survived_p3})

print(df)

df.plot(kind = 'bar', stacked = True)

plt.title("Survived by Pclass")

plt.xlabel("Survived")

plt.ylabel("Count")

plt.show()
# Check Embarked

Survived_S = dataset.loc[dataset.Embarked == 'S', 'Survived'].value_counts()

Survived_C = dataset.loc[dataset.Embarked == 'C', 'Survived'].value_counts()

Survived_Q = dataset.loc[dataset.Embarked == 'Q', 'Survived'].value_counts()



df = pd.DataFrame({'S': Survived_S, "C": Survived_C, "Q": Survived_Q})

df.plot(kind = 'bar', stacked = True)

plt.title("Survived by Embarked")

plt.xlabel("Survived")

plt.ylabel("Count")

plt.show()
label = dataset['Survived']

data = dataset[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

testdata = testset[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]



print(data.shape)

print(data.head())
def fill_nan(data):

    

    data_copy = data.copy(deep = True)

    data_copy['Age'] = data_copy['Age'].fillna(data_copy['Age'].median())

    data_copy['Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())

    data_copy['Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())

    data_copy['Sex'] = data_copy['Sex'].fillna(data_copy['Sex'].mode())

    data_copy['Embarked'] = data_copy['Embarked'].fillna(data_copy['Embarked'].mode().values[0]) # Need to add .values[0] to extract the mode

    

    return data_copy



data_no_nan = fill_nan(data)

testdata_no_nan = fill_nan(testdata)



print(data.isnull().values.any())

print(data_no_nan.isnull().values.any())
# Handle Sex

def transfer_sex(data):

    

    data_copy = data.copy(deep = True)

    data_copy.loc[data_copy.Sex == 'female', 'Sex'] = 0

    data_copy.loc[data_copy.Sex == 'male', 'Sex'] = 1

    

    return data_copy



data_after_sex = transfer_sex(data_no_nan)

testdata_after_sex = transfer_sex(testdata_no_nan)

print(data_after_sex.head())   
# Handle Embarked

def transfer_embark(data):

    

    data_copy = data.copy(deep = True)

    data_copy.loc[data_copy.Embarked == 'S', 'Embarked'] = 0

    data_copy.loc[data_copy.Embarked == 'C', 'Embarked'] = 1

    data_copy.loc[data_copy.Embarked == 'Q', 'Embarked'] = 2

    

    return data_copy



Origin_X_train = transfer_embark(data_after_sex)

X_test = transfer_embark(testdata_after_sex)

print(Origin_X_train.head()) 
from sklearn.model_selection import train_test_split



X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train, label, test_size = 0.2, random_state = 0)



print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



k_range = range(1, 51)

scores = []



for k in k_range:

    

    clf = KNeighborsClassifier(n_neighbors = k)

    clf.fit(X_train, y_train)

    if k % 10 == 0:

        print("k =", k)

    

    y_pred = clf.predict(X_vali)

    scores.append(accuracy_score(y_vali, y_pred))
plt.plot(k_range, scores)

plt.xlabel("K for KNN")

plt.ylabel("Accuracy on Validation Set")

plt.show()
k_best = np.array(scores).argsort()[-1] + 1 # Need to plus ONE because the index starts from 0
clf = KNeighborsClassifier(n_neighbors = k_best)

clf.fit(Origin_X_train, label)

y_pred = clf.predict(X_test)
result = pd.DataFrame({"PassengerId": testset["PassengerId"], "Survived": y_pred})

result.to_csv("submission.csv", index = False, header = True)