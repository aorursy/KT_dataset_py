import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
data_dir = '../input'
dataset = pd.read_csv(data_dir + '/train.csv')
testset = pd.read_csv(data_dir + '/test.csv')
print(dataset.columns)
dataset.head()
dataset.dtypes
dataset.describe()
Survived_male = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_female = dataset.Survived[dataset.Sex == 'female'].value_counts()

df = pd.DataFrame({'male':Survived_male, 'female':Survived_female})
df.plot(kind='bar')
plt.title("survived by sex")
plt.xlabel('survived')
plt.ylabel('count')
plt.show()
dataset['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title('Age distribution')
plt.show()

#查看存活的年龄分布
dataset[dataset.Survived == 1]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who survived")
plt.show()

#查看没存活的年龄分布
dataset[dataset.Survived == 0]['Age'].hist()
plt.ylabel("Number")
plt.xlabel("Age")
plt.title("Age distribution of people who didn't survive")
plt.show()
dataset['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("Number")
plt.title("Fare distribution")
plt.show()

dataset[dataset.Survived == 1]['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("Number")
plt.title("Fare distribution of people who survived")
plt.show()

dataset[dataset.Survived == 0]['Fare'].hist()
plt.xlabel("Fare")
plt.ylabel("Number")
plt.title("Fare distribution of people who didn't survive")
plt.show()
dataset['Pclass'].hist()
plt.xlabel("Pclass")
plt.ylabel("Number")
plt.show()
print(dataset['Pclass'].isnull().values.any())

Survived_p1 = dataset[dataset['Pclass'] == 1]['Survived'].value_counts()
Survived_p2 = dataset[dataset['Pclass'] == 2]['Survived'].value_counts()
Survived_p3 = dataset[dataset['Pclass'] == 3]['Survived'].value_counts()

df = pd.DataFrame({'p1':Survived_p1, 'p2':Survived_p2, 'p3':Survived_p3})
df.plot(kind='bar', stacked=True)
plt.title("survived by pclass")
plt.xlabel("pclass") 
plt.ylabel("count")
plt.show()
Survived_S = dataset.Survived[dataset.Embarked == 'S'].value_counts()
Survived_C = dataset.Survived[dataset.Embarked == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset.Embarked == 'Q'].value_counts()

df = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df.plot(kind = 'bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("Embarked")
plt.ylabel("count")
plt.show()
label = dataset.loc[:, 'Survived']
data = dataset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdat = testset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(data.shape)
print(data.head())
print(testdat.shape)
def fill_NAN(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:, 'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:, 'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:, 'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:, 'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

data_no_nan = fill_NAN(data)
testdat_no_nan = fill_NAN(testdat)

print(data.isnull().values.any())
print(data_no_nan.isnull().values.any())
print(testdat.isnull().values.any())
print(testdat_no_nan.isnull().values.any())

print(data_no_nan)
print(data_no_nan['Sex'].isnull().values.any())

def transfer_sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
testdat_after_sex = transfer_sex(testdat_no_nan)
print(testdat_after_sex)
def transfer_embarked(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embarked(data_after_sex)
testdat_after_embarked = transfer_embarked(testdat_after_sex)
print(testdat_after_embarked)
data_now = data_after_embarked
testdat_now = testdat_after_embarked
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(data_now, label, test_size = 0.2, random_state = 0)

print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

k_range = range(1, 51)
k_scores = []

for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    clf.fit(train_data, train_labels)
    print('K=', K)
    predictions = clf.predict(test_data)
    score = accuracy_score(test_labels, predictions)
    print(score)
    # 检测模型precision， recall 等各项指标
    print(classification_report(test_labels, predictions))
    k_scores.append(score)

plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())
# 预测
clf = KNeighborsClassifier(n_neighbors = 33)
clf.fit(data_now, label)
result = clf.predict(testdat_now)
print(result)
df = pd.DataFrame({"PassengerId": testset['PassengerId'], "Survived": result})
df.to_csv("submission.csv", header=True, index=False)
