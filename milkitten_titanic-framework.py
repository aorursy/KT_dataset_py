import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv') 

print(train_data.columns)

print(test_data.columns)
train_data.head()
print(train_data.dtypes)
train_data.describe()
gp_sex_survived = train_data.groupby(['Sex', 'Survived'])

print(gp_sex_survived.size())

survived_male = train_data.Survived[train_data.Sex=='male'].value_counts()

survived_female = train_data.Survived[train_data.Sex=='female'].value_counts()



df = pd.DataFrame({'male': survived_male, 'female': survived_female})

df.plot(kind='bar', stacked=True, alpha = 0.5)

plt.title('survived by sex')

plt.xlabel('survived')

plt.ylabel('number of people')

plt.show()



# 存活率baed on Sex

survived_Sex = train_data.groupby('Sex')['Survived'].mean()

survived_Sex
survived_age = train_data.Age[train_data.Survived==1].value_counts()

died_age = train_data.Age[train_data.Survived==0].value_counts()

df = pd.DataFrame({'survived': survived_age, 'died': died_age})

df.plot(alpha = 0.5)

plt.title('Age distribution for survival')

plt.xlabel('Age')

plt.ylabel('number')

plt.show()



survived_age = train_data.Age[train_data.Survived==1].value_counts()

died_age = train_data.Age[train_data.Survived == 0].value_counts()

df = pd.DataFrame({'survived': survived_age, 'died': died_age})

df.plot.hist(alpha = 0.5)

plt.title('Age histogram')

plt.xlabel('Age')

plt.ylabel('number')

plt.show()



# train_data.Age.hist()

# train_data.Age[train_data.Survived == 0].hist()

# train_data.Age[train_data.Survived == 1].hist()
survived_fare = train_data.Fare[train_data.Survived==1].value_counts()

died_fare = train_data.Fare[train_data.Survived==0].value_counts()

df = pd.DataFrame({'survived': survived_fare, 'died': died_fare})

df.plot(alpha = 0.5)

plt.title('Fare distribution for survival')

plt.xlabel('Fare')

plt.ylabel('number')

plt.show()



survived_fare = train_data.Fare[train_data.Survived==1]

died_fare = train_data.Fare[train_data.Survived==0]

df = pd.DataFrame({'survived': survived_fare, 'died': died_fare})

df.plot.hist(alpha = 0.5)

plt.title('Fare hist for survival')

plt.xlabel('Fare')

plt.ylabel('number')

plt.show()

# 船票價錢與性別關係

male_survived_fare = train_data.Fare[(train_data.Sex=='male') & (train_data.Survived==1)]

male_died_fare = train_data.Fare[(train_data.Sex=='male') & (train_data.Survived==0)]

female_survived_fare = train_data.Fare[(train_data.Sex=='female') & (train_data.Survived==1)]

female_died_fare = train_data.Fare[(train_data.Sex=='female') & (train_data.Survived==0)]



df = pd.DataFrame({'male_survived': male_survived_fare, 

                   'male_died_fare': male_died_fare,

                   'female_survived': female_survived_fare,

                    'female_died_fare': female_died_fare

                  })

df.plot.hist(alpha = 0.5)

plt.title('Fare hist for Sex')

plt.xlabel('Fare')

plt.ylabel('number')

plt.show()
print(train_data['Pclass'].isnull().values.any())

survived_Pclass_1 = train_data.Survived[train_data.Pclass == 1].value_counts()

survived_Pclass_2 = train_data.Survived[train_data.Pclass == 2].value_counts()

survived_Pclass_3 = train_data.Survived[train_data.Pclass == 3].value_counts()

df = pd.DataFrame({'Class_1': survived_Pclass_1, 'Class_2': survived_Pclass_2, 'Class_3': survived_Pclass_3})

df.plot.bar(stacked=True, alpha = 0.5)

plt.title('Pclass for survival')

plt.xlabel('survived')

plt.ylabel('number')

plt.show()



survived_Pclass = train_data.Pclass[train_data.Survived==1]

died_Pclass = train_data.Pclass[train_data.Survived==0]

df = pd.DataFrame({'survived': survived_Pclass, 'died': died_Pclass})

df.plot.hist(alpha = 0.5)

plt.title('Pclass hist for survival')

plt.xlabel('Pclass')

plt.ylabel('number')

plt.show()
print(train_data['Embarked'].isnull().values.any())

train_data.groupby('Embarked').size()

survived_C = train_data.Survived[train_data.Embarked == 'C'].value_counts()

survived_Q = train_data.Survived[train_data.Embarked == 'Q'].value_counts()

survived_S = train_data.Survived[train_data.Embarked == 'S'].value_counts()

df = pd.DataFrame({'C': survived_C, 'Q': survived_Q, 'S': survived_S})

df.plot.bar(stacked=True, alpha = 0.5)

plt.title('Embarked for survival')

plt.xlabel('survived')

plt.ylabel('number')

plt.show()



embarked_Class_1 = train_data.Embarked[(train_data.Pclass==1)].value_counts()

embarked_Class_2 = train_data.Embarked[(train_data.Pclass==2)].value_counts()

embarked_Class_3 = train_data.Embarked[(train_data.Pclass==3)].value_counts()



df = pd.DataFrame({'Class_1': embarked_Class_1, 'Class_2': embarked_Class_2, 'Class_3': embarked_Class_3})

df.plot.bar(stacked=True, alpha = 0.5)

plt.title('Embarked and Pclass relation')

plt.xlabel('Pclass')

plt.ylabel('number')

plt.show()
train_Y = train_data.Survived

train_x = train_data[['Sex', 'Embarked', 'Pclass', 'Age', 'Fare']]

test_x = test_data[['Sex', 'Embarked', 'Pclass', 'Age', 'Fare']]

print('train y dim', train_Y.shape)

print('train x dim', train_x.shape)

print('test x dim', test_x.shape)
def fill_nan(data):

    data_copy = data.copy(deep=True)

    data_copy.Age = data_copy.Age.fillna(data_copy.Age.median())

    data_copy.Fare = data_copy.Fare.fillna(data_copy.Fare.median())

    data_copy.Sex = data_copy.Sex.fillna('female')

    data_copy.Embarked = data_copy.Embarked.fillna('S')

    data_copy.Pclass = data_copy.Pclass.fillna(data_copy.Pclass.median())

    return data_copy



train_X = fill_nan(train_x)

test_X = fill_nan(test_x)



print(train_x.isnull().values.any())

print(train_X.isnull().values.any())

print(test_x.isnull().values.any())

print(test_X.isnull().values.any())

print(train_Y.isnull().values.any())

print(train_X)
def transfer_sex(data):

    data.loc[data.Sex == 'male', 'Sex'] = 1

    data.loc[data.Sex == 'female', 'Sex'] = 0

    return



transfer_sex(train_X)

print(train_X.head())

transfer_sex(test_X)

print(test_X.head())
def transfer_embarked(data):

    data.loc[data.Embarked == 'S', 'Embarked'] = 0

    data.loc[data.Embarked == 'Q', 'Embarked'] = 1

    data.loc[data.Embarked == 'C', 'Embarked'] = 2

    return



transfer_embarked(train_X)

print(train_X.head())

transfer_embarked(test_X)

print(test_X.head())
# Normalize

def normalize(data):

    data.Embarked = data.Embarked/data.Embarked.max()

    data.Pclass = (data.Pclass - data.Pclass.min())/(data.Pclass.max()-data.Pclass.min())

    data.Age = (data.Age - data.Age.min())/(data.Age.max()-data.Age.min())

    data.Fare = (data.Fare - data.Fare.min())/(data.Fare.max()-data.Fare.min())

    return

normalize(train_X)

print(train_X.head())

normalize(test_X)

print(test_X.head())
from sklearn.model_selection import train_test_split

# split training set 

split_train_X, split_vali_X, split_train_Y, split_vali_Y = train_test_split(train_X, train_Y, random_state=0, test_size=0.2) 

print(split_train_X.shape, split_vali_X.shape, split_train_Y.shape, split_vali_Y.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

K = range(1, 51)

k_score = []

for k in K:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(split_train_X,split_train_Y)

    y_pred = knn.predict(split_vali_X)

    score = accuracy_score(split_vali_Y, y_pred)

    print(k, score)

    k_score.append(score)

plt.plot(K, k_score)

plt.title('k vs score')

plt.xlabel('k')

plt.ylabel('score')

plt.show()

sorted_score = np.array(k_score).argsort()

best_k = sorted_score[-1]+1

print('best_k',best_k)
# 预测

knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(train_X, train_Y)

y_pred = knn.predict(test_X)
df = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

df.to_csv('submission.csv', header=True, index=False)
df = pd.read_csv('submission.csv')

df