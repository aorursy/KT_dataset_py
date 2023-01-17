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

print('Survived male value counts', Survived_m)
print('Survived female value counts', Survived_f)

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
plt.title('Age distribution of dead')
plt.show()

dataset[dataset.Survived == 1]['Age'].hist()
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution of Survived')
plt.show()
dataset['Fare'].hist()
plt.xlabel('Fare')
plt.ylabel('Number')
plt.title('Fare distribution')
plt.show()

dataset[dataset.Survived == 0]['Fare'].hist()
plt.xlabel('Fare')
plt.ylabel('Number')
plt.title('Fare distribution of dead')
plt.show()


dataset[dataset.Survived == 1]['Fare'].hist()
plt.xlabel('Fare')
plt.ylabel('Number')
plt.title('Fare distribution of survived')
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
plt.title('survived by pclass')
plt.xlabel('pclass')
plt.ylabel('count')
plt.show()
dataset['Embarked'].isnull().values.any()
Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()

print(Survived_S)
df = pd.DataFrame({'S': Survived_S, 'C': Survived_C, 'Q': Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title('Survived by embark')
plt.xlabel('Embarked')
plt.ylabel('count')
plt.show()
label = dataset.loc[:, 'Survived']
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
data = dataset.loc[:, features]
testdata = testset.loc[:, features]

print(data.shape)
print(data)
def fill_NAN(data):
    data_copy = data.copy(deep=True)
    for feature in features:
        target = None
        if feature == 'Sex':
            target = 'female'
        elif feature == 'Embarked':
            target = 'S'
        else:
            target = data_copy[feature].median()
        data_copy.loc[:, feature] = target = data_copy[feature].fillna(target)
    return data_copy

data_no_nan = fill_NAN(data)
testdata_no_nan = fill_NAN(testdata)

print(testdata.isnull().values.any())
print(testdata_no_nan.isnull().values.any())
print(data.isnull().values.any())
print(data_no_nan.isnull().values.any())

print(data_no_nan)
def transfer_sex(data):
    data_sex_int = data.copy(deep=True)
    sex_to_int = {'male': 1, 'female': 0}
    data_sex_int['Sex'] = data_sex_int['Sex'].map(sex_to_int)
    return data_sex_int
data_sex_int = transfer_sex(data_no_nan)
testdata_sex_int = transfer_sex(testdata_no_nan)
print(testdata_sex_int)
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
def transfer_embarked(data):
    data_embark_int = data.copy(deep=True)
    y = class_le.fit_transform(data_embark_int['Embarked'].values)
    data_embark_int['Embarked'] = y
    return data_embark_int
data_embark_int = transfer_embarked(data_sex_int)
testdata_embark_int = transfer_embarked(testdata_sex_int)
print(testdata_embark_int)
data_now = data_embark_int
testdata_now = testdata_embark_int
from sklearn.model_selection import train_test_split
train_X, vali_X, train_y, vali_y = train_test_split(data_now, label, random_state=0, test_size=0.2)
print(train_X.shape, vali_X.shape, train_y.shape, vali_y.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1, 51)
k_to_score = {}
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors=K)
    clf.fit(train_X, train_y)
    print('K=', K)
    predictions = clf.predict(vali_X)
    score = accuracy_score(vali_y, predictions)
    print(score)
    k_to_score[K] = score
# 预测
best_k = 1
max_score = 0.0
for k, score in k_to_score.items():
    if score > max_score:
        print(score)
        max_score = score
        best_k = k
print('best k', best_k)
print('score', k_to_score[best_k])

# 检测模型precision， recall 等各项指标
from sklearn.metrics import precision_score, recall_score
clf = KNeighborsClassifier(n_neighbors=best_k)
clf.fit(train_X, train_y)
predictions = clf.predict(vali_X)
print('Precision', precision_score(vali_y, predictions))
print('Recall', recall_score(vali_y, predictions))
# 预测
result = clf.predict(testdata_now)
print(result)
df = pd.DataFrame({'PassengerId': testset['PassengerId'], 'Survived': result})
df.to_csv('submission.csv', header=True, index=False)
