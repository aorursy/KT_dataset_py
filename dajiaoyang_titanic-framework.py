import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
input_dir = '../input/'

data_set = pd.read_csv(input_dir + 'train.csv')

test_set = pd.read_csv(input_dir + 'test.csv')
data_set.columns
data_set.head()
data_set.dtypes
data_set.describe()
Survived_m = data_set.Survived[data_set.Sex == 'male'].value_counts()

Survived_f = data_set.Survived[data_set.Sex == 'female'].value_counts()



df = pd.DataFrame({'male': Survived_m, 'female':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title('Survived by sex')

plt.xlabel('Survived')

plt.ylabel('count')

plt.show()

print(df)
data_set['Age'].hist()

plt.ylabel('Number')

plt.ylim(0, 200)

plt.xlabel('Age')

plt.title('Age distribution')

plt.show()



data_set[data_set.Survived == 0]['Age'].hist()

plt.ylabel('Number')

plt.ylim(0, 200)

plt.xlabel('Age')

plt.title('Age distribution, not survived')

plt.show()



data_set[data_set.Survived == 1]['Age'].hist()

plt.ylabel('Number')

plt.ylim(0, 200)

plt.xlabel('Age')

plt.title('Age distribution, survived')

plt.show()
data_set['Fare'].hist()

plt.ylabel('Number')

plt.ylim(0, 800)

plt.xlabel('Fare Price')

plt.title('Fare distribution')

plt.show()



data_set[data_set.Survived == 0]['Fare'].hist()

plt.ylabel('Number')

plt.ylim(0, 800)

plt.xlabel('Fare Price')

plt.title('Fare distribution, not survived')

plt.show()



data_set[data_set.Survived == 1]['Fare'].hist()

plt.ylabel('Number')

plt.ylim(0, 800)

plt.xlabel('Fare Price')

plt.title('Fare distribution, survived')

plt.show()
data_set['Pclass'].hist()

plt.show()

print(data_set['Pclass'].isnull().values.any())



Survived_p1 = data_set.Survived[data_set['Pclass'] == 1].value_counts()

Survived_p2 = data_set.Survived[data_set['Pclass'] == 2].value_counts()

Survived_p3 = data_set.Survived[data_set['Pclass'] == 3].value_counts()



df = pd.DataFrame({'p1': Survived_p1, 'p2': Survived_p2, 'p3': Survived_p3})



df.plot(kind='bar', stacked=True)

plt.title('Survived by Pclass')

plt.xlabel('Survive')

plt.ylabel('Number')

plt.show()
Survived_S = data_set.Survived[data_set['Embarked'] == 'S'].value_counts()

Survived_C = data_set.Survived[data_set['Embarked'] == 'C'].value_counts()

Survived_Q = data_set.Survived[data_set['Embarked'] == 'Q'].value_counts()



df = pd.DataFrame({'S': Survived_S, 'C': Survived_C, 'Q': Survived_Q})



df.plot(kind='bar', stacked=True)

plt.title('Survived by Embarked')

plt.xlabel('Survive')

plt.ylabel('Number')

plt.show()



print(df)

ratio = df.loc[1, :] / df.sum(axis = 0)

print(ratio)
label = data_set.loc[:, 'Survived']

raw_data = data_set.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

raw_test_data = test_set.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]



print(raw_data.shape)

print(raw_data.head(10))

print(raw_test_data.shape)

print(raw_test_data.head(10))
# set factors



PAR_SEX_FEMALE = 200

PAR_SEX_MALE = 0



PAR_EMBARK_S = 10

PAR_EMBARK_C = 50

PAR_EMBARK_Q = 20



PAR_PCLASS_FACTOR = 30

PAR_FARE_FACTOR = 1

PAR_AGE_FACTOR = 1

def fill_NaN(ori_data):

    data = ori_data.copy(deep=True)

    data.loc[:, 'Age'] = data['Age'].fillna(data['Age'].median()) * PAR_AGE_FACTOR

    data.loc[:, 'Fare'] = data['Fare'].fillna(data['Fare'].median()) * PAR_FARE_FACTOR

    data.loc[:, 'Pclass'] = data['Pclass'].fillna(3) * PAR_PCLASS_FACTOR

    data.loc[:, 'Sex'] = data['Sex'].fillna('male')

    data.loc[:, 'Embarked'] = data['Embarked'].fillna('S')   

    return data



data = fill_NaN(raw_data)

test_data = fill_NaN(raw_test_data)



print(raw_data.isnull().values.any())

print(data.isnull().values.any())

print(raw_test_data.isnull().values.any())

print(test_data.isnull().values.any())



data.head(10)

def transfer_sex(ori_data):

    data = ori_data.copy(deep=True)

    data.loc[data['Sex'] == 'female', 'Sex'] = PAR_SEX_FEMALE

    data.loc[data['Sex'] == 'male', 'Sex'] = PAR_SEX_MALE   

    return data



data_after_sex = transfer_sex(data)

test_data_after_sex = transfer_sex(test_data)

data_after_sex.head(10)
def transfer_embark(ori_data):

    data = ori_data.copy(deep=True)

    data.loc[data['Embarked'] == 'S', 'Embarked'] = PAR_EMBARK_S

    data.loc[data['Embarked'] == 'C', 'Embarked'] = PAR_EMBARK_C

    data.loc[data['Embarked'] == 'Q', 'Embarked'] = PAR_EMBARK_Q 

    return data



data_after_embark = transfer_embark(data_after_sex)

test_data_after_embark = transfer_embark(test_data_after_sex)

print(data_after_embark.head(10))

print(test_data_after_embark.head(10))
from sklearn.model_selection import train_test_split



data_final = data_after_embark

test_data_final = test_data_after_embark



train_data, vali_data, train_label, vali_label = train_test_split(data_final, label, random_state=0, test_size=0.2)

print(train_data.shape, vali_data.shape, train_label.shape, vali_label.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



k_range = range(1, 51)

k_scores = []



for k in k_range:

    print('k = ', k)

    classifier = KNeighborsClassifier(n_neighbors=k)

    classifier.fit(train_data, train_label)

    

    pred_label = classifier.predict(vali_data)

    score = accuracy_score(vali_label, pred_label)

    k_scores.append(score)

    print('{:.3f}'.format(score))

 

#print(k_scores)



plt.plot(k_range, k_scores)

plt.xlabel('k for KNN')

plt.ylabel('Accuracy on validation data set')

plt.show()



k_array = np.array(k_scores)

#print(k_array.argsort())

print('Best k is ' + str(k_array.argmax() + 1) + ' with accuracy of ' + str(k_array.max()))

# 检测模型precision， recall 等各项指标

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(train_data, train_label)

pred_label = classifier.predict(vali_data)



print(classification_report(vali_label, pred_label))

print(confusion_matrix(vali_label, pred_label))
# 预测

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(data_final, label)

result = classifier.predict(test_data_final)



print(result)
df = pd.DataFrame({'PassengerID': test_set['PassengerId'], 'Survived': result})

df.to_csv('submission.csv', header=True, index=False)