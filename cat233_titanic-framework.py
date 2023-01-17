import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
data_dir = '../input/'
dataset = pd.read_csv(data_dir + 'train.csv')
testset = pd.read_csv(data_dir + 'test.csv')
dataset.columns
dataset.head()
print(dataset.dtypes)
print(dataset.describe())
Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()

df = pd.DataFrame({'male': Survived_m, 'female': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title('survived by sex')
plt.xlabel('survived')
plt.ylabel('count')
plt.show()
dataset['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution')
plt.show()

dataset[dataset.Survived == 0]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distributino of people who did not survive')
plt.show()

dataset[dataset.Survived == 1]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution of people who survived')
plt.show()
dataset['Fare'].hist()
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare distribution')
plt.show()

dataset[dataset.Survived == 0]['Fare'].hist()
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare distribution of people who did not survive')
plt.show()

dataset[dataset.Survived == 1]['Fare'].hist()
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare distribution of people who survived')
plt.show()
dataset['Pclass'].hist()
plt.show()
print(dataset['Pclass'].isnull().values.any())

Survived_p1 = dataset.Survived[dataset['Pclass'] == 1].value_counts()
Survived_p2 = dataset.Survived[dataset['Pclass'] == 2].value_counts()
Survived_p3 = dataset.Survived[dataset['Pclass'] == 3].value_counts()

df = pd.DataFrame({'p1': Survived_p1, 'p2': Survived_p2, 'p3': Survived_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title('Survived by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()
Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()

df = pd.DataFrame({'S': Survived_S, 'C': Survived_C, 'Q': Survived_Q})
print(df)
df.plot(kind = 'bar', stacked = True)
plt.title('Survived by Embarked')
plt.xlabel('Survival')
plt.ylabel('Count')
plt.show()
label = dataset.loc[:, 'Survived']
data = dataset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdata = testset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(data.shape)
print(data)
def fill_NAN(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[:, 'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:, 'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:, 'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:, 'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:, 'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

data_no_nan = fill_NAN(data)
testdata_no_nan = fill_NAN(testdata)

print(testdata.isnull().values.any())
print(testdata_no_nan.isnull().values.any())
print(data.isnull().values.any())
print(data_no_nan.isnull().values.any())

print(data_no_nan)
print(data_no_nan['Sex'].isnull().values.any())

def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
testdata_after_sex = transfer_sex(testdata_no_nan)
print(testdata_after_sex)
def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embark(data_after_sex)
testdata_after_embarked = transfer_embark(testdata_after_sex)
print(testdata_after_embarked)
# 训练数据
data_now = data_after_embarked
testdata_now = testdata_after_embarked

from sklearn.model_selection import train_test_split

train_data, val_data, train_labels, val_labels = train_test_split(data_now,label,test_size=0.2,random_state=0)
print(train_data.shape, val_data.shape, train_labels.shape, val_labels.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
k_range = range(1, 51)
k_scores = []
for k in k_range:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_data, train_labels)
    print('k=',k)
    predictions = clf.predict(val_data)
    score = accuracy_score(val_labels, predictions)
    print(score)
    print(classification_report(val_labels, predictions))
    print(confusion_matrix(val_labels, predictions))
    k_scores.append(score)
plt.plot(k_range, k_scores)
plt.xlabel('k for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())
# 预测
clf = KNeighborsClassifier(n_neighbors=33)
clf.fit(data_now,label)
result = clf.predict(testdata_now)
print(result)
df = pd.DataFrame({'PassengerId': testset['PassengerId'],
                  'Survived': result})
df.to_csv('submission.csv',header=True,index=False)
