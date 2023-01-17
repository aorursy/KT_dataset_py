import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
titanic_train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission= pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
titanic_train_data.head()
titanic_train_data.isnull().sum()
titanic_train_data = titanic_train_data.drop(columns='Cabin')

titanic_test_data = titanic_test_data.drop(columns='Cabin')
median_male_age = titanic_train_data[titanic_train_data.Sex == "male"].Age.median()

median_female_age = titanic_train_data[titanic_train_data.Sex == "female"].Age.median()
titanic_train_data.Age = np.where((titanic_train_data.Age.isnull()) & (titanic_train_data.Sex == 'female'), median_female_age, titanic_train_data.Age) 

titanic_train_data.Age = np.where((titanic_train_data.Age.isnull()) & (titanic_train_data.Sex == 'male'), median_male_age, titanic_train_data.Age) 



titanic_test_data.Age = np.where((titanic_test_data.Age.isnull()) & (titanic_test_data.Sex == 'female'), median_male_age, titanic_test_data.Age) 

titanic_test_data.Age = np.where((titanic_test_data.Age.isnull()) & (titanic_test_data.Sex == 'male'), median_male_age, titanic_test_data.Age) 
titanic_train_data.Embarked.fillna(titanic_train_data.Embarked.mode()[0],inplace=True)



titanic_test_data.Embarked.fillna(titanic_test_data.Embarked.mode()[0],inplace=True)
titanic_train_data.isnull().sum()
titanic_train_data.Sex.value_counts().plot(kind='bar')
sns.countplot(x="Survived",hue="Sex", data=titanic_train_data)
sns.catplot(x="Pclass", y="Survived", hue="Sex", data=titanic_train_data, kind="bar")
titanic_train_data.Embarked.value_counts().plot(kind='bar')
titanic_train_data.hist(column='Parch')
ranges = {'Age':[],'people_in_range':[],'survived_count':[]}



for age in range(0,80,5):

    range_text = '{}-{}'.format(age,age+5)

    people_in_range = titanic_train_data.query('(Age > @age) & (Age < @age + 5)')

    people_in_range_count = people_in_range.PassengerId.count()

    survived_count =  people_in_range.query('Survived == 1').PassengerId.count()

    

    ranges['Age'].append(range_text)

    ranges['people_in_range'].append(people_in_range_count)

    ranges['survived_count'].append(survived_count)

    

ranges = pd.DataFrame(ranges)
ranges.head()
plt.figure(figsize=(20,8))

plt.title('People in age range')

sns.barplot(x=ranges.Age, y=ranges.people_in_range)
plt.figure(figsize=(20,8))

plt.title('Survived people in age range')

sns.barplot(x=ranges.Age, y=ranges.survived_count)
titanic_train_data['Solo'] = (titanic_train_data.Parch == 0).apply(int)

titanic_test_data['Solo'] = (titanic_test_data.Parch == 0).apply(int)
titanic_train_data.Solo.value_counts()
titanic_train_data['Title'] = titanic_train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

titanic_test_data['Title'] = titanic_test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



titanic_train_data.Title.value_counts()
titanic_train_data.Title = titanic_train_data.Title.replace(['Capt', 'Col', 'Dr', 'Major', 'Rev', 'Don', 'Sir', 'Jonkheer'], 'Mr')

titanic_train_data.Title = titanic_train_data.Title.replace(['Ms', 'Mlle'], 'Miss')

titanic_train_data.Title = titanic_train_data.Title.replace(['Mme', 'Lady', 'Countess', 'Dona'], 'Mrs')



titanic_test_data.Title = titanic_test_data.Title.replace(['Capt', 'Col', 'Dr', 'Major', 'Rev', 'Don', 'Sir', 'Jonkheer'], 'Mr')

titanic_test_data.Title = titanic_test_data.Title.replace(['Ms', 'Mlle'], 'Miss')

titanic_test_data.Title = titanic_test_data.Title.replace(['Mme', 'Lady', 'Countess', 'Dona'], 'Mrs')



titanic_train_data.Title.value_counts()
title_map = {'Mr':0,'Miss':1,'Mrs':2,'Master':3}

titanic_train_data.Title = titanic_train_data.Title.map(title_map)

titanic_test_data.Title = titanic_test_data.Title.map(title_map)
titanic_train_data.head()
titanic_train_data = titanic_train_data.drop(columns=['Ticket', 'PassengerId','Name'])

titanic_test_data = titanic_test_data.drop(columns=['Ticket', 'PassengerId','Name'])
titanic_train_data.head()
titanic_train_data.Sex = np.where(titanic_train_data.Sex == 'male', 1, 0)

titanic_test_data.Sex = np.where(titanic_test_data.Sex == 'male', 1, 0)



titanic_train_data.head()
Embarked_map = {'Q':0,'S':1,'C':2}

titanic_train_data.Embarked = titanic_train_data.Embarked.map(Embarked_map)

titanic_test_data.Embarked = titanic_test_data.Embarked.map(Embarked_map)



titanic_train_data.head()
def numeric_to_catigorial(column,bins):

    #get ranges

    ranges = pd.cut(column,bins=bins)

    ranges = np.unique(ranges)

    # convert to list for speed

    column = list(column)



    for variable_value, variable_rage in enumerate(ranges):

        for i, item in enumerate(column):

            if column[i] in variable_rage:

                column[i] = variable_value

    return column
titanic_train_data.Age = numeric_to_catigorial(titanic_train_data.Age,4)

titanic_test_data.Age = numeric_to_catigorial(titanic_test_data.Age,4)



titanic_train_data.Fare = numeric_to_catigorial(titanic_train_data.Age,4)

titanic_test_data.Fare = numeric_to_catigorial(titanic_test_data.Age,4)
titanic_train_data.head()
from sklearn.model_selection import train_test_split
X = titanic_train_data.drop(columns='Survived')

y = titanic_train_data.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
accuracies = {}
rf_clf = RandomForestClassifier()
params = {

    'n_estimators': range(10,1001,50),

    'max_depth': range(1,13,2),

    'min_samples_leaf': range(1,30),

    'min_samples_split': range(2,50,2)

}
search = RandomizedSearchCV(rf_clf,params,cv=10,n_jobs=-1)

search.fit(X_train,y_train)

best_rf = search.best_estimator_



y_pred_rf = best_rf.predict(X_test)
accuracies['random_forest'] = round(best_rf.score(X_test,y_test)*100,2)

cm_rf = confusion_matrix(y_test,y_pred_rf)



print('Random forest accuracy is: {}%'.format(accuracies['random_forest']))
from sklearn.neighbors import KNeighborsClassifier
k = 2

knn = KNeighborsClassifier(n_neighbors = k) 

knn.fit(X_train, y_train)



y_pred_knn = knn.predict(X_test)
accuracies['knn_score'] = round(knn.score(X_test,y_test)*100,2)

cm_knn = confusion_matrix(y_test,y_pred_knn)



print("{}NN accuracy: {}%".format(k,accuracies['knn_score']))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)

lr.fit(X_train,y_train)



y_pred_lr = lr.predict(X_test)
accuracies['lr_score'] = round(lr.score(X_test,y_test)*100,2)

cm_lr = confusion_matrix(y_test,y_pred_lr)



print("Logistic regression accuracy: {}%".format(accuracies['lr_score']))
sns.set_style("whitegrid")



plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Random forest Confusion Matrix.")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,3)

plt.title("Logistic regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
sns.set_style("whitegrid")

plt.figure(figsize=(10,5))

plt.title('Accuracy')

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
accuracies
submit = best_rf.predict(titanic_test_data)
pd.DataFrame({'PassengerId':gender_submission.PassengerId,'Survived':submit}).to_csv('SurvivedPrediction.csv',index=False)