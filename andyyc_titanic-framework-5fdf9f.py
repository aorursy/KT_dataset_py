import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库
trainSet = pd.read_csv('../input/train.csv')
testSet = pd.read_csv('../input/test.csv')

testSet
trainSet.head()
print(trainSet.dtypes)

print(trainSet.describe())

Survived_male = trainSet['Survived'][trainSet['Sex'] == 'male'].value_counts()
Survived_female = trainSet['Survived'][trainSet['Sex'] == 'female'].value_counts()
Survived_male
Survived_female
df = pd.DataFrame({'male': Survived_male, 'female': Survived_female})
df
df.plot(kind='bar', stacked=True)
plt.xlabel('Sex')
plt.ylabel('# of People')
trainSet['Age'].hist()
trainSet[trainSet.Survived == 0]['Age'].hist()
trainSet[trainSet.Survived == 1]['Age'].hist()
plt.xlabel("Age")
plt.ylabel("# of People")
plt.legend(["All", "Dead", "Alive"])
trainSet['Fare'].hist(density=True)
plt.ylabel("Number")
plt.xlabel("Fare")
plt.title('Fare distribution')
plt.show()

trainSet[trainSet.Survived == 0]['Fare'].hist(density=True)
plt.ylabel("Number %")
plt.xlabel("Fare")
plt.title('Fare distribution of people who did not survive')
plt.show()

trainSet[trainSet.Survived == 1]['Fare'].hist(density=True)
plt.ylabel("Number %")
plt.xlabel("Fare")
plt.title('Fare distribution of people who survived')
plt.show()
trainSet['Pclass'].hist()
# Survived_p1 = trainSet['Survived'][trainSet.Pclass == 1].value_counts()
Survived_p1 = trainSet.loc[:, 'Survived'][trainSet.Pclass == 1].value_counts()
Survived_p2 = trainSet['Survived'][trainSet.Pclass == 2].value_counts()
Survived_p3 = trainSet['Survived'][trainSet.Pclass == 3].value_counts()
df = pd.DataFrame({'p1': Survived_p1, 'p2': Survived_p2, 'p3': Survived_p3})  # Think of as adding columns
df
df.plot(kind='bar', stacked=True)
plt.title("survived by pclass")
plt.xlabel("pclass")
plt.ylabel("count")
plt.show()

trainSet['Embarked']
Survived_C = trainSet.loc[:, 'Survived'][trainSet['Embarked'] == 'C'].value_counts()
Survived_S = trainSet['Survived'][trainSet['Embarked'] == 'S'].value_counts()
Survived_Q = trainSet['Survived'][trainSet['Embarked'] == 'Q'].value_counts()
df = pd.DataFrame({'S': Survived_S, 'C': Survived_C, 'Q': Survived_Q})
df
df.plot(kind='bar', stacked=True)
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.show()
train_label = trainSet.loc[:, 'Survived']
# pclass, sex, age, fare, embarked
train_data = trainSet.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
test_data = testSet.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
train_data.dtypes
train_data['Age'].isnull().any()
def fill_NAN(df):
    df_cp = df.copy(deep=True)
    df_cp.loc[:, 'Pclass'] = df_cp['Pclass'].fillna(df_cp['Pclass'].median())
    df_cp.loc[:, 'Sex'] = df_cp['Sex'].fillna('female')
    df_cp.loc[:, 'Age'] = df_cp['Age'].fillna(df_cp['Age'].median())
    df_cp.loc[:, 'Fare'] = df_cp['Fare'].fillna(df_cp['Fare'].median())
    df_cp.loc[:, 'Embarked'] = df_cp['Embarked'].fillna('S')
    return df_cp

train_data_noNAN = fill_NAN(train_data)
test_data_noNAN = fill_NAN(test_data)
print(train_data_noNAN.isnull().values.any())
print(train_data_noNAN['Sex'].isnull().any())
def transfer_sex(data):
    data1 = data.copy(deep=True)
    data1.loc[data1.Sex == 'female', 'Sex'] = 0;
    data1.loc[data1.Sex == 'male', 'Sex'] = 1;
    return data1

train_data_sex = transfer_sex(train_data_noNAN)
test_data_sex = transfer_sex(test_data_noNAN)
def transfer_embark(data):
    data1 = data.copy(deep=True)
    data1.loc[data1['Embarked'] == 'S', 'Embarked'] = 0
    data1.loc[data1['Embarked'] == 'C', 'Embarked'] = 1
    data1.loc[data1['Embarked'] == 'Q', 'Embarked'] = 2
    return data1

train_data_embark = transfer_embark(train_data_sex)
test_data_embark = transfer_embark(test_data_sex)
print(train_data_embark.isnull().any()) # no more null data
print(test_data_embark.isnull().any())
train_data_embark.shape
test_data_embark.shape
trainSet['Survived'].shape
# x_data = train_data_embark
x_test = test_data_embark

from sklearn.model_selection import train_test_split

x_data, x_vali, y_data, y_vali = train_test_split(train_data_embark, train_label, test_size = 0.2, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_range = range(1, 51)
k_scores = []

for k in k_range:
    # Train model
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_data, y_data)
    print('K = ', k)

    # Validation
    y_pred = knn.predict(x_vali)
    accuracy = accuracy_score(y_pred, y_vali)
    print(accuracy)
    k_scores.append(accuracy)
# cross validation 找到最好的k值
plt.plot(k_range, k_scores)
plt.xlabel('The k number for KNN')
plt.ylabel('Accuracy Score')
plt.title('Validation result from training data')
np.array(k_scores).argsort() + 1
# 预测
clf = KNeighborsClassifier(n_neighbors = 33)
clf.fit(x_data, y_data) # should we use all data to fit the model, instead of just 80%?
y_test = clf.predict(x_test)

df = pd.DataFrame({"PassengerId": testSet['PassengerId'], "Survived": y_test})
df.to_csv('submission.csv',header=True)
print('DONE')
