import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data.columns
data.head()
data.dtypes
data.info()
data.describe()
data.head()
print(data['Sex'].head())
survived_m = data['Survived'][data['Sex'] == 'male'].value_counts()
survived_f = data['Survived'][data['Sex'] == 'female'].value_counts()

df = pd.DataFrame({'Male': survived_m, 'Female': survived_f})
df.plot(kind='bar', stacked=True)
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Survival by Sex')
plt.show()

data.Age.head()
data.Age.hist()
plt.show()
data.Age.isnull().values.any()
plt.scatter(data.Survived, data.Age)
plt.xlabel('Survival')
plt.ylabel('Age')
plt.title('Age by Survived')
plt.show()
data.Fare.head()
data.Fare.hist()
plt.show()
data.Fare.isnull().values.any()
plt.scatter(data.Survived, data.Fare)
plt.show()
data.Pclass.head()
data.Pclass.isnull().values.any()
data.Pclass.hist()
plt.show()
survived_P1 = data['Survived'][data['Pclass'] == 1].value_counts()
survived_P2 = data['Survived'][data['Pclass'] == 2].value_counts()
survived_P3 = data['Survived'][data['Pclass'] == 3].value_counts()

df = pd.DataFrame({'P1': survived_P1, 'P2': survived_P2, 'P3': survived_P3})
df.plot(kind='bar', stacked=True)
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Survived by Pclass')
plt.show()
data.Embarked.head()
data.Embarked.value_counts()
data.Embarked.isnull().values.any()
survived_S = data['Survived'][data['Embarked'] == 'S'].value_counts()
survived_C = data['Survived'][data['Embarked'] == 'C'].value_counts()
survived_Q = data['Survived'][data['Embarked'] == 'Q'].value_counts()

df = pd.DataFrame({'Southampton':survived_S, 'Cherbourg':survived_C, 'Queenstown':survived_Q})
df.plot(kind='bar', stacked=True)
plt.legend(loc='best')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Survived by Embarked')
plt.show()
data.head()
label = data[['Survived']]
training_data = data[['Pclass','Sex','Age','Fare','Embarked']]
testing_data = test[['Pclass','Sex','Age','Fare','Embarked']]
print(training_data.shape)
print(testing_data.shape)
def fill_NAN(data):
    data_copy = data.copy(deep=True)
    data_copy['Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy['Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy['Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy['Sex'] = data_copy['Sex'].fillna('female')
    data_copy['Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy
training_no_nan = fill_NAN(training_data)
testing_no_nan = fill_NAN(testing_data)
print(training_data.isnull().values.any(), training_no_nan.isnull().values.any(), sep='\t')
print(testing_data.isnull().values.any(), testing_no_nan.isnull().values.any(), sep='\t')
training_no_nan.head()
def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy['Sex'][data_copy['Sex'] == 'female'] = 0
    data_copy['Sex'][data_copy['Sex'] == 'male'] = 1
    return data_copy

training_after_sex = transfer_sex(training_no_nan)
testing_after_sex = transfer_sex(testing_no_nan)
print(training_after_sex.Sex.value_counts())
def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy['Embarked'][data_copy['Embarked'] == 'S'] = 0
    data_copy['Embarked'][data_copy['Embarked'] == 'C'] = 1
    data_copy['Embarked'][data_copy['Embarked'] == 'Q'] = 2
    return data_copy

training_after_embark = transfer_embark(training_after_sex)
testing_after_embark = transfer_embark(testing_after_sex)
print(training_after_embark.head())
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
training_set = training_after_embark
testing_set = testing_after_embark
X_train, X_val, y_train, y_val = train_test_split(training_set.values,
                                                 label.values.ravel(),
                                                 test_size=0.2,
                                                 random_state=0)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('Accuracy: ', accuracy_score(y_val, y_pred))
print('Confusion Matrix: ', confusion_matrix(y_val, y_pred), sep='\n')
print('Classification Report: ', classification_report(y_val, y_pred), sep='\n')

from sklearn.model_selection import cross_val_score
k_range = range(1,40)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, training_set, label.values.ravel(), cv=5, scoring='accuracy')
    print('k = ' + str(k) + ', scores = ' + str(scores) + '\n mean = ' + str(scores.mean()))
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('K-Value')
plt.ylabel('Accuracy')
plt.title('Accyracy by K')
plt.show()
plt.plot([20,21,22,23,24,25], k_scores[20:26])
plt.show()
clf = KNeighborsClassifier(n_neighbors=21)
clf.fit(training_set, label.values.ravel())
y_test = clf.predict(testing_set)
df = pd.DataFrame({"PassengerId": test['PassengerId'],"Survived": y_test})
df.to_csv('gender_submission.csv', header=True)