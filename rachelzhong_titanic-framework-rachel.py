import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')
dataset.columns
dataset.head()
print(dataset.dtypes)
print(dataset.describe())
Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()
dat = pd.DataFrame({'male': Survived_m, 'female': Survived_f})
dat.plot(kind = 'bar', stacked = True)
plt.title('Survival by sex')
plt.xlabel('Survival')
plt.ylabel('Count')
plt.show()
dataset['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Number')
plt.title('Age Distribution')
plt.show()

dataset[dataset.Survived == 0]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age Distribution of people who did not survive')
plt.show()

dataset[dataset.Survived == 1]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age Distribution of people who did survive')
plt.show()

dataset['Fare'].hist()
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare Distribution')
plt.show()

dataset[dataset.Survived == 0]['Fare'].hist()
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare Distribution of people who did not survive')
plt.show()

dataset[dataset.Survived == 1]['Fare'].hist()
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare Distribution of people who did survive')
plt.show()
dataset['Pclass'].hist()
plt.show()

Survived_p1 = dataset.Survived[dataset['Pclass'] == 1].value_counts()
Survived_p2 = dataset.Survived[dataset['Pclass'] == 2].value_counts()
Survived_p3 = dataset.Survived[dataset['Pclass'] == 3].value_counts()

df = pd.DataFrame({'p1': Survived_p1, 'p2': Survived_p2, 'p3': Survived_p3})
df.plot(kind = 'bar', stacked = True)
plt.title('Survival by pclass')
plt.xlabel('Survival')
plt.ylabel('count')
plt.show()
Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()

df = pd.DataFrame({'S': Survived_S, 'C': Survived_C, 'Q': Survived_Q})
df.plot(kind = 'bar', stacked = True)
plt.title('Survival by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.show()
label = dataset.loc[:, 'Survived']
data = dataset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdat = testset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
def fill_NAN(data):
    data_copy = data.copy(deep = True)
    data_copy['Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy['Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy['Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy['Sex'] = data_copy['Sex'].fillna('female')
    data_copy['Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

data_no_nan = fill_NAN(data)
testdat_no_nan = fill_NAN(testdat)
print(data_no_nan.isnull().values.any())
def transfer_sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy
data_after_sex = transfer_sex(data_no_nan)
testdat_after_sex = transfer_sex(testdat_no_nan)
def transfer_embark(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0 # loc is to access a gropu of rows
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embark(data_after_sex)
testdat_after_embarked = transfer_embark(testdat_after_sex)
data_now = data_after_embarked
testdat_now = testdat_after_embarked
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data_now, label, random_state = 0, test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_scores = []
k_range = range(1, 51)
for k in k_range:
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(train_data, train_labels)
    print('k=', k)
    predictions = clf.predict(test_data)
    score = accuracy_score(test_labels, predictions)
    print(score)
    k_scores.append(score)
plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())
# 预测
clf = KNeighborsClassifier(n_neighbors = 33)
clf.fit(data_now, label)
# 检测模型precision， recall 等各项指标
result = clf.predict(testdat_now)

# 预测


df = pd.DataFrame({'PassengerId': testset['PassengerId'], 'Survived': result})
df.to_csv('submission.csv', header = True)
