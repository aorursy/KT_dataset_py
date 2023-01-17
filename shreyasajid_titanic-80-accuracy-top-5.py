#Exploratory Data Analysis and Wrangling
import pandas as pd
import numpy as np
import random as rnd
#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
#Read the csv file
df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")
#Combine train and test data
combine = [df_train, df_test]
#Dimensions of train
df_train.shape
#Dimensions of test
df_test.shape
#Obtain Feature Names
print(df_train.columns.values)
#Overview of train
df_train.head()
#Overview of test
df_test.head()
#Description of train
df_train.describe()
#Description of test
df_test.describe()
#Check for missing values in train
df_train.isnull().sum()
#Check for missing values in test
df_test.isnull().sum()
#To check the datatypes of the features in train
df_train.info()
#To check the datatypes of the features in test
df_test.info()
#Frequency table for Sex
df_train['Sex'].value_counts()
#Frequency table for Embarked
df_train['Embarked'].value_counts()
#Frequency table for Ticket
df_train['Ticket'].value_counts()
#Frequency table for Name
df_train['Name'].value_counts()
#Frequency table for Cabin
df_train['Cabin'].value_counts()
df_train.describe(include=['O'])
#To determine how representative is the training dataset of the actual problem domain
df_train.Survived.value_counts(normalize=True)
df_train.Parch.value_counts(normalize=True)
df_train.SibSp.value_counts(normalize=True)
#Boxplot - Age
plt.figure(figsize = (8,3))
sns.boxplot(x = 'Age',data = df_train,color = "pink")
plt.title("Age of the passenger")
#Histogram (distribution analysis) - Age of the passenger
df_train['Age'].hist(bins = 30)
plt.title("Age of the passenger")
#Boxplot - Fare of the ticket
plt.figure(figsize = (10,5))
sns.distplot(df_train['Age'].dropna(),kde=True,color = 'green')
plt.title("Fare of the ticket")
#As subplots
fig, axes = plt.subplots(2,4, figsize=(16, 10))
sns.countplot('Survived',data=df_train,ax=axes[0,0])
sns.countplot('Pclass',data=df_train,ax=axes[0,1])
sns.countplot('Sex',data=df_train,ax=axes[0,2])
sns.countplot('SibSp',data=df_train,ax=axes[0,3])
sns.countplot('Parch',data=df_train,ax=axes[1,0])
sns.countplot('Embarked',data=df_train,ax=axes[1,1])
sns.distplot(df_train['Fare'], kde=True,ax=axes[1,2])
sns.distplot(df_train['Age'].dropna(),kde=True,ax=axes[1,3])
g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.9, aspect=1.6) 
grid.map(plt.hist, 'Age', alpha=.5, bins=20) 
grid.add_legend();
grid = sns.FacetGrid(df_train, col='Pclass', hue='Survived')
#grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.9, aspect=1.6) 
grid.map(plt.hist, 'Age', alpha=.5, bins=20) 
grid.add_legend();
grid = sns.FacetGrid(df_train, row='Embarked', size=2.2, aspect=1.6) 
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep') 
grid.add_legend()
grid = sns.FacetGrid(df_train, col='Embarked') 
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep') 
grid.add_legend()
grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', size=2.2, aspect=1.6) 
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None,color = 'purple') 
grid.add_legend()
grid = sns.FacetGrid(df_train, col='Embarked', hue='Survived', palette={0: 'k', 1: 'g'})  
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None) 
grid.add_legend()
print("Before", df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)
#Correcting by dropping features
df_train = df_train.drop(['Ticket'], axis=1)
df_test = df_test.drop(['Ticket'], axis=1)
#Combine test and train
combine = [df_train, df_test]
print("After", df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)
#Extract new feature using regular expressions
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(df_train['Title'], df_train['Sex'])
#Replace less frequent titles as 'Others'
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')
#Replace typos or less common title with more frequently used titles
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#To combine Parch and SibSp into a single variable 
combine = [df_train, df_test]
for dataset in combine:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    #Creating a new variable called travelled_alone
    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'
    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'
#Drop Parch, SibSp
df_train = df_train.drop(['Parch', 'SibSp'], axis=1)
df_test = df_test.drop(['Parch', 'SibSp'], axis=1)
combine = [df_train, df_test]
#Fill the missing values with random numbers computed based on mean and the standard deviation of the column.
#for dataset in combine:
 #   mean = df_train["Age"].mean()
 #  std = df_train["Age"].std()
 # is_null = dataset["Age"].isnull().sum()
 # compute random numbers between the mean, std and is_null
 # rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
   # age_slice = dataset["Age"].copy()
   # age_slice[np.isnan(age_slice)] = rand_age
   # dataset["Age"] = age_slice
   # dataset["Age"] = df_train["Age"].astype(int)
df_train['Embarked'].value_counts()
for dataset in combine:
    dataset['Embarked'].fillna('S',inplace = True)
#Check for missing values
df_train.isnull().sum()
df_test.isnull().sum()
df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)
df_test.head()
for dataset in combine:
    dataset['Cabin_ID'] = dataset['Cabin'].str[0]
for dataset in combine:
    dataset['Cabin_ID'].fillna('O',inplace = True)
df_train['Cabin_ID'].value_counts()
#Correcting by dropping features
df_train = df_train.drop(['Cabin'], axis=1)
df_test = df_test.drop(['Cabin'], axis=1)
df_train.dtypes
#Convert the categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Others": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
df_train.head()
#Drop the Name feature from train and test
#Drop PassengerId feature from train
df_train = df_train.drop(['Name', 'PassengerId'], axis=1)
df_test = df_test.drop(['Name'], axis=1)
combine = [df_train, df_test]
df_train.shape, df_test.shape
#Import Library
from sklearn.preprocessing import LabelEncoder
#LABEL ENCODING
le = LabelEncoder()
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ) 
#Convert to numericals
for dataset in combine:
    dataset['travelled_alone'] = le.fit_transform(dataset['travelled_alone'].astype(str))

### No - 0, Yes - 1 
df_train.head()
#Convert fare from float to numerical
#df_train['Fare'] = df_train['Fare'].astype(int)
#df_test['Fare'] = df_test['Fare'].astype(int)
for dataset in combine:
    dataset['Cabin_ID'] = le.fit_transform(dataset['Cabin_ID'].astype(str))
df_train.dtypes
for dataset in combine:
    dataset =pd.get_dummies(dataset,columns=['Cabin_ID'])
guess_ages = np.zeros((2,3)) 
guess_ages 
#CLEANING
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_mean = guess_df.mean()
            #age_std = guess_df.std()
            age_median = guess_df.median()
            age_guess = rnd.uniform(age_median - age_mean, age_median + age_mean)

            
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16.136, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16.136) & (dataset['Age'] <= 32.102), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32.102) & (dataset['Age'] <= 48.068), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48.068) & (dataset['Age'] <= 64.034), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64.034, 'Age']
df_train.head()
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

df_train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
df_train.isnull().sum()
df_test.isnull().sum()
X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop("PassengerId", axis=1).copy()
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("Score = ",acc_log)
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)  
Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("Score = ",acc_knn)
gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train)  
Y_pred = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print("Score = ",acc_knn)
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print("Score = ",acc_svc)
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  
Y_pred = decision_tree.predict(X_test)  

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("Score = ",acc_decision_tree)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Score = ",acc_random_forest)
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN','Naive Bayes','Support Vector Machines','Decision Tree','Random Forest'],
    'Score': [acc_log, acc_knn, acc_gaussian, acc_svc, acc_decision_tree, acc_random_forest]})
df_result = results.sort_values(by='Score', ascending=False)
#Display
df_result
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'Feature':X_train.columns,'Importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('Importance',ascending=False)
importances
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Score = ",acc_random_forest)
#Create param grid object 
forest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)
#Instantiate Random Forest model
forest = RandomForestClassifier()
#Build and fit model 
forest_cv = GridSearchCV(estimator=forest, param_grid=forest_params, cv=5) 
forest_cv.fit(X_train, Y_train)
print("Best score: {}".format(forest_cv.best_score_))
print("Optimal params: {}".format(forest_cv.best_estimator_))
Y_pred = forest_cv.predict(X_test)
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)
print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))
submission = pd.DataFrame({"PassengerId": df_test["PassengerId"],"Survived": Y_pred})
#submission.to_csv('../output/submission.csv', index=False)
