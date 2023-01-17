import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/train.csv')

testset = pd.read_csv('../input/test.csv')
dataset.columns
dataset.dtypes
dataset.head()
dataset.describe()
survived_m = dataset.Survived[dataset.Sex =='male'].value_counts()

survived_f = dataset.Survived[dataset.Sex =='female'].value_counts()

df = pd.DataFrame({'male': survived_m, 'female': survived_f})

df.plot(kind = 'bar', stacked = True)

plt.title('Survived by sex')

plt.xlabel('Survived')

plt.ylabel('Count')

plt.show()
dataset['Age'].hist()

plt.title('Age Distribution')

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()



dataset[dataset.Survived == 1]['Age'].hist()

plt.title('Age Distribution by people who survived')

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()



dataset[dataset.Survived == 0]['Age'].hist()

plt.title('Age Distribution by people who did not survive')

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()
dataset['Fare'].hist()

plt.title('Ticket Distribution')

plt.xlabel('Ticket Price')

plt.ylabel('Count')

plt.show()



dataset[dataset.Survived == 1]['Fare'].hist()

plt.title('Ticket Distribution by people who survived')

plt.xlabel('Ticket Price')

plt.ylabel('Count')

plt.show()



dataset[dataset.Survived == 0]['Fare'].hist()

plt.title('Ticket Distribution by people who did not survive')

plt.xlabel('Ticket Price')

plt.ylabel('Count')

plt.show()
Unsurvived_class = dataset.Pclass[dataset.Survived == 0].value_counts()

Survived_class= dataset.Pclass[dataset.Survived == 1].value_counts()

df = pd.DataFrame({'Survive': Survived_class, 'Unsurvived': Unsurvived_class})

df.plot(kind = 'bar', stacked = True)

plt.title('Survived vs Unsurvived by Pclass')

plt.xlabel('Pclass')

plt.ylabel('Count')

plt.show()
Unsurvived_loc = dataset.Embarked[dataset.Survived == 0].value_counts()

survived_loc = dataset.Embarked[dataset.Survived == 1].value_counts()

df = pd.DataFrame({'Survived': survived_loc, 'Unsurvived_loc': Unsurvived_loc})

df.plot(kind='bar', stacked = True)

plt.title('Survived vs Unsurvived by Embarked')

plt.xlabel('Embarked')

plt.ylabel('count')

plt.show()
label = dataset.loc[:, 'Survived']

data = dataset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

testdata = testset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]



data.head(10)
testdata.head(10)
def fill_NaN(data):

    data_copy = data.copy(deep=True)

    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())

    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())

    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())

    data_copy.loc[:, 'Sex'] = data_copy['Sex'].fillna('male')

    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')

    

    return data_copy



data_no_nan = fill_NaN(data)

testdata_no_nan = fill_NaN(testdata)



print(testdata.isnull().values.any())

print(data.isnull().values.any())

print(testdata_no_nan.isnull().values.any())

print(data_no_nan.isnull().values.any())

def transfer_sex(data):

    data_copy = data.copy(deep=True)

    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0

    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1

    

    return data_copy



data_after_sex = transfer_sex(data_no_nan)

testdata_after_sex = transfer_sex(testdata_no_nan)



print(data_after_sex)
def transfer_embarked(data):

    data_copy = data.copy(deep = True)

    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1

    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2

    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 3

    return data_copy



data_after_embarked = transfer_embarked(data_after_sex)

testdata_after_embarked = transfer_embarked(testdata_after_sex)

data_after_embarked.head(10)
data_processed = data_after_embarked

testdata_processed = testdata_after_embarked



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
train_data, val_data, train_labels, val_labels = train_test_split(data_processed, label, random_state = 0, test_size = 0.2)



print(train_data.shape)

print(val_data.shape)
k_range = range(1, 51)

k_scores = []



for K in k_range:

    clf = KNeighborsClassifier(n_neighbors= K)

    clf.fit(train_data, train_labels)

    print('K= ', K)

    predictions = clf.predict(val_data)

    score = accuracy_score(val_labels, predictions)

    print(score)

    k_scores.append(score)
plt.plot(k_range, k_scores)

plt.xlabel('K for KNN')

plt.ylabel('Accuracy on validation set')

plt.show()



print(np.array(k_scores).argsort())
# 预测

clf = KNeighborsClassifier(n_neighbors = 22)

clf.fit(data_processed, label)

result = clf.predict(testdata_processed)



print(result)
df = pd.DataFrame({'PassengerId': testset['PassengerId'], 'Survived': result})

df.to_csv('submission.csv', header = True, index = False)