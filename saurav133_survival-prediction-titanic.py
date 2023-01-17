import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline



#load training data

train = pd.read_csv('../input/train.csv');



train.head(10)
#load test data

test = pd.read_csv('../input/test.csv');



test.head(10)
complete_set = [train, test]



for data in complete_set:

    print(pd.isnull(data).sum())

    print('........................')
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
print(train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
for data in complete_set:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1



print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for data in complete_set:

    data['IsAlone'] = 0

    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1



print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for data in complete_set:

    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])



print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for data in complete_set:

    data['Fare'] = data['Fare'].fillna(train['Fare'].median())

    

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)



print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for data in complete_set:

    age_avg = data['Age'].mean()

    age_std = data['Age'].std()

    age_null_count = data['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)

    

    data['Age'][np.isnan(data['Age'])] = age_null_random_list

    data['Age'] = data['Age'].astype(int)

    

train['CategoricalAge'] = pd.cut(train['Age'], 5)



print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
for data in complete_set:

    data['Title'] = ''

    data['Title']=data['Name'].str.extract('([A-Za-z]+)\.')

    

    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    

    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')



print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()) 

#train.head()
for data in complete_set:

    #Mapping dummies for Sex

    data['Sex'] = data['Sex'].map({'female':0, 'male':1}).astype(int)

    

    #Mapping dummies for Titles

    title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}

    data['Title'] = data['Title'].map(title_mapping)

    data['Title'] = data['Title'].fillna(0)

    

    #Mapping dummies for Embarked

    data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2})

    

    #Mapping dummies for Fare

    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2

    data.loc[data['Fare'] > 31, 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(float)

    

    #Mapping dummies for Age

    data.loc[data['Age'] <= 16, 'Age'] = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[data['Age'] > 64, 'Age'] = 4

    

#Feature Selection

drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']

train.drop(drop_elements, axis=1, inplace=True)

train.drop(['CategoricalAge', 'CategoricalFare', 'PassengerId'], axis=1, inplace=True)

test.drop(drop_elements, axis=1, inplace=True)

    
train.head()
test.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.40, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))

print('\n')

print(accuracy_score(y_test,predictions))
from sklearn.naive_bayes import MultinomialNB



NBModel = MultinomialNB().fit(X_train,y_train)



predictions = NBModel.predict(X_test)



print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))

print('\n')

print(accuracy_score(y_test,predictions))
from sklearn.svm import SVC



model = SVC()



model.fit(X_train, y_train)



predictions = model.predict(X_test)



print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))

print('\n')

print(accuracy_score(y_test,predictions))
from sklearn.model_selection import GridSearchCV



param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}



grid = GridSearchCV(SVC(), param_grid, verbose=3)



grid.fit(X_train, y_train)
grid_predections = grid.predict(X_test)



print(confusion_matrix(y_test, grid_predections))

print('\n')

print(classification_report(y_test, grid_predections))

print('\n')

print(accuracy_score(y_test,grid_predections))
#test_temp = test.drop('PassengerId', axis=1)

#predictions = grid.predict(test_temp)

#df = pd.concat([test['PassengerId'], pd.DataFrame(predictions, columns= ['Survived'])], axis=1)

#df.to_csv('Survival', index=False)



#train.head()
test.head()
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=200)



rfc.fit(X_train, y_train)



rfc_pred = rfc.predict(X_test)



print(confusion_matrix(y_test, rfc_pred))

print('\n')

print(classification_report(y_test, rfc_pred))

print('\n')

print(accuracy_score(y_test,rfc_pred))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train, y_train)



knn_pred = knn.predict(X_test)



print(confusion_matrix(y_test, knn_pred))

print('\n')

print(classification_report(y_test, knn_pred))

print('\n')

print(accuracy_score(y_test,knn_pred))
error_rate = []



for i in range (1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))



plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate vs K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=13)



knn.fit(X_train, y_train)



knn_pred = knn.predict(X_test)



print(confusion_matrix(y_test, knn_pred))

print('\n')

print(classification_report(y_test, knn_pred))