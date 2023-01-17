import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

trainset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/train.csv')
trainset.columns
trainset.head()
print(trainset.dtypes)
print(trainset.describe())
survived_m = trainset.Survived[trainset.Sex == 'male'].value_counts()
survived_f = trainset.Survived[trainset.Sex == 'female'].value_counts()

df = pd.DataFrame({'male': survived_m, 'female': survived_f})
df.plot(kind='bar', stacked=True)
plt.title('Survived by sex')
plt.xlabel('survived')
plt.ylabel('count')
plt.show()
age = trainset['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Number')
plt.title('Age Distribution')
plt.show()

age = trainset[trainset.Survived == 0]['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Number')
plt.title('Not Survived Age Distribution')
plt.show()

age = trainset[trainset.Survived == 1]['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Number')
plt.title('Survived Age Distribution')
plt.show()

fare = trainset['Fare'].hist()
plt.xlabel('Fare')
plt.ylabel('Number')
plt.title('Fare Distribution')
plt.show()

fare = trainset[trainset.Survived == 0]['Fare'].hist()
plt.xlabel('Fare')
plt.ylabel('Number')
plt.title('Not Survived Age Distribution')
plt.show()

fare = trainset[trainset.Survived == 1]['Fare'].hist()
plt.xlabel('Fare')
plt.ylabel('Number')
plt.title('Survived Age Distribution')
plt.show()
trainset['Pclass'].hist()
plt.show()

# 查NaN
print(trainset['Pclass'].isnull().values.any())

survived_p1 = trainset.Survived[trainset['Pclass'] == 1].value_counts()
survived_p2 = trainset.Survived[trainset['Pclass'] == 2].value_counts()
survived_p3 = trainset.Survived[trainset['Pclass'] == 3].value_counts()

df = pd.DataFrame({'p1': survived_p1, 'p2': survived_p2, 'p3':survived_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title("survived by pclass")
plt.xlabel("pclass") 
plt.ylabel("count")
plt.show()
survived_s = trainset.Survived[trainset['Embarked'] == 'S'].value_counts()
survived_q = trainset.Survived[trainset['Embarked'] == 'Q'].value_counts()
survived_c = trainset.Survived[trainset['Embarked'] == 'C'].value_counts()

print(survived_s)
df = pd.DataFrame({'S':survived_s, 'Q':survived_q, 'C':survived_c})
df.plot(kind='bar', stacked=True)
plt.title("Survived by Embarked")
plt.xlabel("Survival") 
plt.ylabel("count")
plt.show()
# trainset.loc[行，列]
label = trainset.loc[:, 'Survived']
data=trainset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdata=testset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(data.shape)
def fill_NaN(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[:, 'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:, 'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:, 'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:, 'Embarked'] = data_copy['Embarked'].fillna('S')
    data_copy.loc[:, 'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    return data_copy

data_no_nan = fill_NaN(data)
test_no_nan = fill_NaN(testdata)

    
def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

data_after_transfer_sex = transfer_sex(data_no_nan)
test_data_after_transfer_sex = transfer_sex(test_no_nan)

def transfer_embarked(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_ready = transfer_embarked(data_after_transfer_sex)
test_ready = transfer_embarked(test_data_after_transfer_sex)
print(test_ready)
# 80/20 分 trainset and validation set
from sklearn.model_selection import train_test_split
train_data, vali_data, train_labels, vali_labels = train_test_split(data_ready, label, random_state=0, test_size=0.2)
print(train_data.shape, vali_data.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1, 51)
k_scores = list()
for k in k_range:
    classfier = KNeighborsClassifier(n_neighbors=k)
    classfier.fit(train_data, train_labels)
    print('k=', k)
    predictions = classfier.predict(vali_data)
    score = accuracy_score(vali_labels, predictions)
    print(score)
    k_scores.append(score)
# find the best k
plt.plot(k_range, k_scores)
plt.xlabel('K')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())
# 预测, K = 33, 0 base的不是32, result 对应每个 test set里的passenger id和survived
final_classifier = KNeighborsClassifier(n_neighbors=33)
final_classifier.fit(data_ready, label)
result = final_classifier.predict(test_ready)
print(result)
df = pd.DataFrame({'PassengerId': testset['PassengerId'], 'Survived': result})
df.to_csv('submission.csv',header=True, index=False)