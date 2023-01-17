import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

trainset = pd.read_csv('../input/train.csv')

testset = pd.read_csv('../input/test.csv')
trainset.columns
print(trainset.head())
print(trainset.dtypes)
print(trainset.describe())
survived_f = trainset.Survived[trainset['Sex'] == 'female'].value_counts()

survived_m = trainset.Survived[trainset['Sex'] == 'male'].value_counts()



df = pd.DataFrame({'female': survived_f, 'male': survived_m})

df.plot(kind='bar', stacked=True)

plt.title('Survived by sex')

plt.xlabel('Survived')

plt.ylabel('Count')

plt.show()
trainset['Age'].hist()

plt.title('Age distribution')

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()



trainset.Age[trainset['Survived'] == 1].hist()

plt.title('Age distribution of people who survived')

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()



trainset.Age[trainset['Survived'] == 0].hist()

plt.title('Age distribution of people who did not survive')

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()
trainset['Fare'].hist()

plt.title('Fare distribution')

plt.xlabel('Fare')

plt.ylabel('Count')

plt.show()



trainset.Fare[trainset['Survived'] == 1].hist()

plt.title('Fare distribution of people who survived')

plt.xlabel('Fare')

plt.ylabel('Count')

plt.show()



trainset.Fare[trainset['Survived'] == 0].hist()

plt.title('Fare distribution of people who did not survive')

plt.xlabel('Fare')

plt.ylabel('Count')

plt.show()
trainset['Pclass'].hist()

plt.title('Class distribution')

plt.xlabel('Class')

plt.ylabel('Count')

plt.show()



p1 = trainset.loc[trainset['Pclass'] == 1, 'Survived'].value_counts()

p2 = trainset.loc[trainset['Pclass'] == 2, 'Survived'].value_counts()

p3 = trainset.loc[trainset['Pclass'] == 3, 'Survived'].value_counts()



df = pd.DataFrame({'p1': p1, 'p2': p2, 'p3': p3})

df.plot(kind='bar', stacked=True)

plt.xlabel('Survived')

plt.ylabel('Count')

plt.show()
embarked_C = trainset.loc[trainset['Embarked'] == 'C', 'Survived'].value_counts()

embarked_Q = trainset.loc[trainset['Embarked'] == 'Q', 'Survived'].value_counts()

embarked_S = trainset.loc[trainset['Embarked'] == 'S', 'Survived'].value_counts()



df = pd.DataFrame({'C': embarked_C, 'Q': embarked_Q, 'S': embarked_S})

df.plot(kind='bar', stacked=True)

plt.xlabel('Survived')

plt.ylabel('Count')

plt.show()
data = trainset[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

label = trainset['Survived']

testdata = testset[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
data.isnull().sum()
def fill_NAN(data):

    data_no_nan = data.copy(deep=True)

    data_no_nan['Pclass'] = data_no_nan['Pclass'].fillna(data_no_nan['Pclass'].median())

    data_no_nan['Age'] = data_no_nan['Age'].fillna(data_no_nan['Age'].median())

    data_no_nan['Fare'] = data_no_nan['Fare'].fillna(data_no_nan['Fare'].median())

    data_no_nan['Sex'] = data_no_nan['Sex'].fillna('male')

    data_no_nan['Embarked'] = data_no_nan['Embarked'].fillna('S')

   

    return data_no_nan



data_no_nan = fill_NAN(data)

testdata_no_nan = fill_NAN(testdata)



print(data.isnull().values.any())

print(data_no_nan.isnull().values.any())

print(testdata.isnull().values.any())

print(testdata_no_nan.isnull().values.any())
def convert_sex(data):

    data_copy = data.copy(deep=True)

    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0

    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1

    return data_copy



data_after_sex = convert_sex(data_no_nan)

testdata_after_sex = convert_sex(testdata_no_nan)

print(testdata_after_sex)
def convert_embark(data):

    data_copy = data.copy(deep=True)

    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 0

    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 1

    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 2

    return data_copy



data_after_embark = convert_embark(data_after_sex)

testdata_after_embark = convert_embark(testdata_after_sex)

print(testdata_after_embark)
data_after_embark[['Age', 'Fare']].describe()
def scale(data):

    data_copy = data.copy(deep=True)

    data_copy['Age'] = (data_copy['Age'] - 0.42) / (80.0 - 0.42)

    data_copy['Fare'] = (data_copy['Fare'] - 0.) / (512.3292 - 0.)

    return data_copy



data_after_scale = scale(data_after_embark)

testdata_after_scale = scale(testdata_after_embark)



print(testdata_after_scale)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



data_now, test_data_now = data_after_scale, testdata_after_scale



# split data into train and validation

train_X, valid_X, train_y, valid_y = train_test_split(data_now, label, \

                                                      test_size=0.2, random_state=43)



print(train_X.shape, valid_X.shape, train_y.shape, valid_y.shape)
k_range = range(1, 101)

accuracy = []



for k in k_range:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(train_X, train_y)

    pred_y = model.predict(valid_X)

    score = accuracy_score(valid_y, pred_y)

    accuracy.append(score)

    print('k = %s, accuracy score: %.5f' % (k, score))
plt.plot(k_range, accuracy)

plt.title('Accuracy score vs k')

plt.xlabel('k')

plt.ylabel('Accuracy')

plt.show()



np.array(accuracy).argsort()
# 预测

best_k = 12

best_model = KNeighborsClassifier(n_neighbors=best_k)

best_model.fit(train_X, train_y)

pred_valid_y = best_model.predict(valid_X)
# 检测模型precision， recall 等各项指标

from sklearn.metrics import precision_score, recall_score, f1_score



print('precision score: %s' % (precision_score(valid_y, pred_valid_y)))

print('recall score: %s' % (recall_score(valid_y, pred_valid_y)))

print('f1 score: %s' % (f1_score(valid_y, pred_valid_y)))
# 预测

pred_y = best_model.predict(test_data_now)

print(pred_y)
pd.DataFrame({'PassengerId': testset['PassengerId'], 'Survived': pred_y}).to_csv('titanic_submission.csv', header=True, index=False)