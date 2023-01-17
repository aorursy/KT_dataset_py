import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
titanic = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")
# Creating a list with two files, more accuracy for to the math to fill NaN values
combined = [titanic, titanic_test]
titanic.head()
titanic_test.head()
#Checking if exist some NaN value
len(titanic[titanic['Sex'].isnull()])
#How many unique enters this array have.
titanic["Sex"].unique()
#Checking which gender have more survivers
titanic[['Survived', 'Sex']].groupby('Sex').mean()
# Replacing Categorical variables by continuous, with this for and this list(combined),
# we can replace Sex in titanic and test_ticanic

dicsex = {"male": 0, "female": 1}
for dfsex in combined:
    dfsex['Sex'] = dfsex['Sex'].map(dicsex)
    
#other method    
#titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
#titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic.head()
#Checking if exist some NaN value
len(titanic[titanic['Embarked'].isnull()])
titanic[titanic['Embarked'].isnull()]
# Trying to found an insight, connecting all common variables of these people to predict where they embarked, without success
titanic[(titanic['Pclass'] == 1) & (titanic['Survived'] == 1) & (titanic['Sex'] == 1)].groupby('Embarked').sum()
# Here we can see, at all classes most of people embarked on "S", so to fill this data with less variation we put "S".
sns.countplot(x = 'Pclass', data = titanic, hue = 'Embarked')
titanic["Embarked"] = titanic["Embarked"].fillna("S")
# Same concept to replace used at Sex column
dic_embarked = {"S": 0, "C": 1, "Q": 2}
for df_embarked in combined:
    df_embarked['Embarked'] = df_embarked['Embarked'].map(dic_embarked)
# getting all Title from Name column in both DataFrames through the list and creating a new column with those titles
for df in combined:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=True)
# concat both updated dataframes to see the distribution of titles at titanic
combined_df = pd.concat([titanic, titanic_test], axis = 0)
combined_df['Title'].value_counts()
plt.subplots(figsize = (16,6))
sns.countplot(x = 'Title', data = combined_df)
# Same concept to replace used at Sex column, considering the 4 largest groups of people and the rest of them in 1 group
# Mr: 0
# Miss: 1
# Mrs: 2
# Master: 3
# Others: 4
titlemap = {"Mr": 0,
            "Miss": 1, 
            "Mrs": 2, 
            "Master": 3, 
            "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,"Countess": 4, "Ms": 4, 
            "Lady": 4, "Jonkheer": 4,"Don": 4, "Dona" : 4, "Mme": 4,"Capt": 4,"Sir": 4 }
for df in combined:
    df['Title'] = df['Title'].map(titlemap)
titanic.head()
# Lets see how is the distribuition by gender for people who survived and whos dont.
survived = titanic[titanic['Survived']==1]['Sex'].value_counts()
# Extract how many peoples for each sex survived
dead = titanic[titanic['Survived']==0]['Sex'].value_counts()
# Extract how many peoples for each sex not survived
df = pd.DataFrame([survived,dead])
df.columns= ['Male', 'Female']
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=1)
#looking for null values ate Age.
len(titanic[titanic['Age'].isnull()])
# Here we need to make a decision, which  variable we'll relate with Age to fill the empty values
titanic[titanic['Age'].isnull()].head()
# looking to the first possibility, calculate new ages through Pclass
combined_df[['Age','Pclass']].groupby('Pclass').mean()
#Here we have more accuracy, title it is very related to age
combined_df = pd.concat([titanic, titanic_test], axis = 0)
combined_df[['Age','Title']].groupby('Title').mean()
# this is one function to verify all null values at AGE, and substitute to this respective new value considering Title
# using combined_df, we can be more accurate to get the new mean values, because we consider a bigger data
def impute_age(cols):
    Age = cols[0]
    Title = cols[1]
    
    if pd.isnull(Age):
        if Title == 0:
            return combined_df['Age'][combined_df['Title'] == 0].mean()
        elif Title == 1:
            return combined_df['Age'][combined_df['Title'] == 1].mean()
        elif Title == 2:
            return combined_df['Age'][combined_df['Title'] == 2].mean()
        elif Title == 3:
            return combined_df['Age'][combined_df['Title'] == 3].mean()
        else:
            return combined_df['Age'][combined_df['Title'] == 4].mean()
    else:
        return Age
# aplly the function on DF's, titanic and titanic_test
titanic['Age'] = titanic[['Age','Title']].apply(impute_age,axis=1)
titanic_test['Age'] = titanic_test[['Age','Title']].apply(impute_age,axis=1)
# To improve our machine learning model, we need to smooth our data, so we'll divide our Age values in 5 categories
for df_age in combined:
    df_age.loc[ df_age['Age'] <= 16, 'Age'] = 0,
    df_age.loc[(df_age['Age'] > 16) & (df_age['Age'] <= 26), 'Age'] = 1,
    df_age.loc[(df_age['Age'] > 26) & (df_age['Age'] <= 36), 'Age'] = 2,
    df_age.loc[(df_age['Age'] > 36) & (df_age['Age'] <= 62), 'Age'] = 3,
    df_age.loc[ df_age['Age'] > 62, 'Age'] = 4
    #using this for, we can substitute all values at titanic and also titanic_test
# Lets see how is the distribuition by Age for people who survived and whos dont.
survivedAge = titanic[titanic['Survived']==1]['Age'].value_counts()
# Extract how many peoples for each Age survived
deadAge = titanic[titanic['Survived']==0]['Age'].value_counts()
# Extract how many peoples for each Age not survived
df = pd.DataFrame([survivedAge,deadAge])
df.columns= ['0-16','17-26', '27-36', '37-62', '63+']
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=1)
df_age2 = titanic[['Age', 'Survived']]
sns.countplot(x = 'Age', hue = 'Survived', data = df_age2)
titanic_test.head()
#looking for null values ate Fare, Test DF.
titanic_test[titanic_test['Fare'].isnull()]
#looking for null values ate Fare, Train DF.
titanic[titanic['Fare'].isnull()]
# In general, the fare paid is directly relate to the class. the miss value was replaced by the mean value at third class
titanic_test['Fare'] = titanic_test['Fare'].fillna(combined_df['Fare'][combined_df['Pclass'] == 3].mean())
combined_df = pd.concat([titanic, titanic_test], axis = 0)
combined_df['Fare'].describe()
# Isn't good to see the distribution in that way
# To improve our machine learning model, we need to smooth our data, so we'll divide our Fare values in 4 categories
for dataset in combined:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
titanic_test.head()
for dfcabin in combined:
    dfcabin['Cabin'] = dfcabin['Cabin'].str[:1]
# if we try to associate Cabin location to Fare, data are very confused
print(titanic[titanic['Fare'] == 0]['Cabin'].unique())
print(titanic[titanic['Fare'] == 1]['Cabin'].unique())
print(titanic[titanic['Fare'] == 2]['Cabin'].unique())
print(titanic[titanic['Fare'] == 3]['Cabin'].unique())
# if we try to associate Cabin location to Pclass, we can see a pattern
print(titanic[titanic['Pclass'] == 1]['Cabin'].unique())
print(titanic[titanic['Pclass'] == 2]['Cabin'].unique())
print(titanic[titanic['Pclass'] == 3]['Cabin'].unique())

PC1 = titanic[titanic['Pclass'] == 1]['Cabin'].value_counts()
PC2 = titanic[titanic['Pclass'] == 2]['Cabin'].value_counts()
PC3 = titanic[titanic['Pclass'] == 3]['Cabin'].value_counts()
dfCabin = pd.DataFrame([PC1, PC2, PC3])
dfCabin.index = ['1 Class', '2 Class', '3 Class']
dfCabin.plot(kind = 'bar', stacked = True, figsize = (12,6))
plt.style.use('bmh')
# Due Cabin A,B,T,C only exist at First class, they become only A
titanic['Cabin'].replace(['B', 'T', 'C'], ['A', 'A', 'A'], inplace = True);
titanic_test['Cabin'].replace(['B', 'T', 'C'], ['A', 'A', 'A'], inplace = True);
titanic['Cabin'].unique()
dicCabins = {"A": 0, "D": 0.5, "E": 1, "F": 1.5, "G": 2}
for dataset2 in combined:
    dataset2['Cabin'] = dataset2['Cabin'].map(dicCabins)
def impute_cabin(cols):
    Cabin  = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Cabin):
        if Pclass == 1:
            return 0
        elif Pclass == 2:
            return 1
        else:
            return 1.5
    else:
        return Cabin
titanic['Cabin'] = titanic[['Cabin','Pclass']].apply(impute_cabin,axis=1)
titanic_test['Cabin'] = titanic[['Cabin','Pclass']].apply(impute_cabin,axis=1)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=100, shuffle=True, random_state=0)
X_train = titanic.drop(["Survived",'PassengerId', 'Name', 'Ticket'], axis=1)
Y_train = titanic["Survived"]
X_test  = titanic_test.drop(["PassengerId", 'Name', 'Ticket'], axis=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 25)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
score_knn = cross_val_score(knn, X_train, Y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy')
print('KNN Cross: {}\nKNN:       {}'.format(round(np.mean(score_knn)*100,3), acc_knn))
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X_train, Y_train)
Y_pred = logistic.predict(X_test)
acc_log = round(logistic.score(X_train, Y_train) * 100, 2)
score_lr = cross_val_score(logistic, X_train, Y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy')
print('Logistic Cross: {}\nLogistic:       {}'.format(round(np.mean(score_lr)*100,2), acc_log))
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
score_dt = cross_val_score(decision_tree, X_train, Y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy')
print('Decision Tree Cross: {}\nDecision Tree:       {}'.format(round(np.mean(score_dt)*100,2), acc_decision_tree))
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
score_rf = cross_val_score(random_forest, X_train, Y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy')
print('Random Forest Cross: {}\nRandom Forest:       {}'.format(round(np.mean(score_rf)*100,2), acc_random_forest))
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(X_train, Y_train)
Y_pred = gbk.predict(X_test)
acc_gbk = round(gbk.score(X_train, Y_train) * 100, 2)
score_gbk = cross_val_score(gbk, X_train, Y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy')
print('Gradient Boosting Classifier Cross: {}\nGradient Boosting Classifier:       {}'.format(round(np.mean(score_gbk)*100,2), acc_gbk))
from sklearn.svm import SVC

svc = SVC(gamma = 'scale')
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
score_svc = cross_val_score(svc, X_train, Y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy')
print('SVC Cross: {}\nSVC:       {}'.format(round(np.mean(score_svc)*100,2), acc_svc))
from sklearn.model_selection import train_test_split
x_trainG, x_testeG, y_trainG, y_testeG = train_test_split(titanic.drop(["Survived",'PassengerId',
                                                                    'Name', 'Ticket', 'Cabin'],
                                                                     axis=1), titanic["Survived"], 
                                                                     test_size = 0.4, random_state = 101)
error_rate = []
for i in range(1,35):
    knnG = KNeighborsClassifier(n_neighbors = i)
    knnG.fit(x_trainG, y_trainG)
    y_predG = knnG.predict(x_testeG)
    error_rate.append(np.mean(y_predG!=y_testeG))
plt.figure(figsize = (14, 8))
plt.plot(range(1, 35), error_rate, color = 'blue', ls = 'dashed', marker = 'o')

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree',
              'Random Forest', 'KNN', 'Gradient Boosting Classifier', 'SVC'],
    'Score': [np.mean(score_lr)*100, np.mean(score_dt)*100, np.mean(score_rf)*100,
             np.mean(score_knn)*100, np.mean(score_gbk)*100, np.mean(score_svc)*100]})
print('Cross Validation')
models.sort_values(by='Score', ascending=False)
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree',
              'Random Forest', 'KNN', 'Gradient Boosting Classifier', 'SVC'],
    'Score': [acc_log, acc_decision_tree, acc_random_forest,
             acc_knn, acc_gbk, acc_svc]})
print('Score')
models.sort_values(by='Score', ascending=False)
# Run the Model First
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)