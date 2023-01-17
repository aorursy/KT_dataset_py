import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
train.info()
test.info()
train.describe()
train.head()
train.isnull().any()
test.isnull().any()
train.Age.isna().value_counts()
mean_age_all = (train.Age.mean() + test.Age.mean())/2
train.Age.fillna(mean_age_all, inplace=True)
test.Age.fillna(mean_age_all, inplace=True)

median_fare_all = (train.Fare.median() + test.Fare.median())/2
test.Fare.fillna(median_fare_all, inplace=True )
train.Embarked.value_counts(), test.Embarked.value_counts()
train.Embarked.isna().value_counts()
train.Embarked.fillna( 'S', inplace=True )
train.drop(columns=['Cabin'], axis=1, inplace=True)
test.drop(columns=['Cabin'], axis=1, inplace=True)
train.isnull().any().any() == test.isnull().any().any() == False
train.drop(['Ticket'], axis=1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
gender_encoder.fit(train.Sex)
train.Sex = gender_encoder.transform(train.Sex)
test.Sex = gender_encoder.transform(test.Sex)
embarked_encoder = LabelEncoder()
embarked_encoder.fit(train.Embarked)
train.Embarked = embarked_encoder.transform(train.Embarked)
test.Embarked  = embarked_encoder.transform(test.Embarked)
def NameProcess(name):
    if name.find('Mrs') != -1:
        return 'Mrs'
    if name.find('Mr') != -1:
        return 'Mr'
    if name.find('Miss') != -1:
        return 'Miss'
    if name.find('Master') != -1:
        return 'Master'
    return 'Normal'
train.Name = train.Name.map(NameProcess)
test.Name  = test.Name.map(NameProcess)

name_encoder = LabelEncoder()
name_encoder.fit(train.Name)
train.Name = name_encoder.transform(train.Name)
test.Name  = name_encoder.transform(test.Name)
train.head()
import seaborn as sns
sns.heatmap(train.corr())
train.head()
train.Fare.hist()
sns.boxplot(x=train.Fare)
Q1 = pd.Series(np.hstack((train.Fare.values, test.Fare.values))).quantile(0.25)
Q3 = pd.Series(np.hstack((train.Fare.values, test.Fare.values))).quantile(0.75)
IQR = Q3 - Q1
print(IQR)
train_fare_filter = ((train.Fare < (Q1 - 1.5 * IQR)) | (train.Fare > (Q3 + 1.5 * IQR)))
test_fare_filter = ((test.Fare < (Q1 - 1.5 * IQR)) | (test.Fare > (Q3 + 1.5 * IQR)))
train_fare_filter.value_counts(), test_fare_filter.value_counts()
train.Fare[train_fare_filter] = (train.Fare.mean() + test.Fare.mean())/2
test.Fare[test_fare_filter] = (train.Fare.mean() + test.Fare.mean())/2
train.head()
sns.boxplot(x=train.Fare)
train.Fare = np.array(pd.cut(train.Fare, bins=5, labels=[1,2,3,4,5]))
test.Fare = np.array(pd.cut(test.Fare, bins=5, labels=[1,2,3,4,5]))
train.head()
sns.boxplot(train.Age)
train.Age = np.array(pd.cut(train.Age, bins=5, labels=[1,2,3,4,5]))
test.Age = np.array(pd.cut(test.Age, bins=5, labels=[1,2,3,4,5]))
train = pd.get_dummies(train, columns=['Name', 'Age', 'Fare', 'Sex', 'Pclass', 'Embarked'])
test = pd.get_dummies(test, columns=['Name', 'Age', 'Fare', 'Sex', 'Pclass', 'Embarked'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
train['Family'] = train.SibSp+train.Parch+1
test['Family'] = test.SibSp+test.Parch+1
train = train.drop(['Parch', 'SibSp'], axis=1)
test = test.drop(['Parch', 'SibSp'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    train.drop(columns=['PassengerId', 'Survived'], axis=1).values, 
    train.Survived.values,
    test_size=0.25, 
    random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
accuracy_score(y_test, clf.predict(X_test))
accuracy_score(y_train, clf.predict(X_train))
prediction = clf.predict(test.drop('PassengerId', axis=1))
result = pd.DataFrame({'PassengerId':test.PassengerId , 'Survived':prediction})
result.head()
pd.read_csv('../input/gender_submission.csv').head()
result.to_csv('final1.csv', index=False)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
X_train, X_test, y_train, y_test = train_test_split(
    train.drop(columns=['PassengerId', 'Survived'], axis=1).values, 
    to_categorical(train.Survived.values),
    test_size=0.25,
    random_state=42)
model = Sequential([
    Dense(128, input_dim=X_train.shape[1]),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(16),
    Activation('relu'),
    Dense(8),
    Activation('sigmoid'),
    Dense(2)
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=100)
prediction = np.argmax(model.predict(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)
accuracy_score(y_test, prediction)
prediction = np.argmax(model.predict(test.drop('PassengerId', axis=1)), axis=1)
result = pd.DataFrame({'PassengerId':test.PassengerId , 'Survived':prediction})
result.to_csv('laaaast.csv', index=False)
result.head(10)







test.head()




























































