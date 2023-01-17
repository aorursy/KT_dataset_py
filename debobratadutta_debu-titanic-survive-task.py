# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read Titanic Training Data

titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_train.head()
#Read Titanic Test Data

titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')

titanic_test.head()
#Import Visualiton Library

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')
# Lets Clean the Data First

# Analyze the Null entries

# Visualize the Null entries in Training DataSet

sns.heatmap(titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Visualize the Null entries in Test DataSet

sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Most of the entries in Cabin data is empty . Droping Cabin column from both the dataset

# Droping PassengerId and Ticket as well

titanic_train.drop(['PassengerId','Cabin', 'Ticket'],axis=1,inplace=True)

titanic_test.drop(['PassengerId','Cabin', 'Ticket'],axis=1,inplace=True)
# Check the Null Data Visualization again

sns.heatmap(titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Check Age Vs PClass Box Plot

sns.boxplot(x='Pclass',y='Age',data=titanic_train)
# Filling in median age, mode embark, and mediam fare for missing values

concact_df = [titanic_train, titanic_test]

for df in concact_df:    

    #Filling missing age with median

    df['Age'].fillna(df['Age'].median(), inplace = True)



    #Filling embarked with mode

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)



    #Filling missing fare with median

    df['Fare'].fillna(df['Fare'].median(), inplace = True)
sns.heatmap(titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.heatmap(titanic_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(8,12))

sns.swarmplot(x="Sex", y="Age", hue="Survived", data=titanic_train, palette="bright").set_title(' Age wise Survive')
# Above graph is showing Female is more survive in between Age group 25 to 30

# Check all possible values for Embarked column

# we should covert categorical features to numerical values (Sex, Embarked)



titanic_train['Sex'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

titanic_train['Embarked'].replace(to_replace=['C','Q', 'S'], value=[0,1,2],inplace=True)



titanic_test['Sex'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

titanic_test['Embarked'].replace(to_replace=['C','Q', 'S'], value=[0,1,2],inplace=True)
#Set Title from Name

titanic_train['Title'] = titanic_train['Name'].str.split(",", expand=True)[1].str.split(".", expand=True)[0]

titanic_test['Title'] = titanic_test['Name'].str.split(",", expand=True)[1].str.split(".", expand=True)[0]



titanic_train['Title']=titanic_train['Title'].apply(lambda x: x[1:])

titanic_test['Title']=titanic_test['Title'].apply(lambda x: x[1:])



titanic_train['Title'] = titanic_train['Title'].replace('Mlle', 'Miss')

titanic_train['Title'] = titanic_train['Title'].replace('Ms', 'Miss')

titanic_train['Title'] = titanic_train['Title'].replace('Mme', 'Mrs')

titanic_train['Title'] = titanic_train['Title'].replace(['Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')



titanic_test['Title'] = titanic_test['Title'].replace('Mlle', 'Miss')

titanic_test['Title'] = titanic_test['Title'].replace('Ms', 'Miss')

titanic_test['Title'] = titanic_test['Title'].replace('Mme', 'Mrs')

titanic_test['Title'] = titanic_test['Title'].replace(['Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')



titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



titanic_train['Title'] = titanic_train['Title'].map(titles)

titanic_test['Title'] = titanic_test['Title'].map(titles)

   

# filling NaN with 0, to get safe

titanic_train['Title'] = titanic_train['Title'].fillna(0)

titanic_test['Title'] = titanic_test['Title'].fillna(0)
# Check Different Visualtion for Survived vs Embarked Category(S-1,C-2,Q-3)

fig, ax = plt.subplots(1, 2, figsize=(12,6))

left = sns.countplot(x='Embarked',data = titanic_train, ax=ax[0]).set_title('Embarked Category Passengers count')

right = sns.countplot(x='Embarked',data = titanic_train[titanic_train['Survived']==1], ax=ax[1]).set_title('Embarked Category Survive count')
# Check Different Visualtion for Survived (0 - Male and 1 - Female)

fig, ax = plt.subplots(1, 2, figsize=(12,6))

left = sns.countplot(x='Sex',data = titanic_train, ax=ax[0]).set_title('Gender Wise Passengers count')

right = sns.countplot(x='Sex',data = titanic_train[titanic_train['Survived']==1], ax=ax[1]).set_title('Gender Wise Survive count')
# In Above graph we can see Female are survived more than male

# Check the Survived Vs Pclass and Sex

fig, ax = plt.subplots(1, 2, figsize=(12,6))

left = sns.countplot(x='Pclass',data = titanic_train, hue='Sex', ax=ax[0]).set_title('P-Class Wise Passengers count')

right = sns.countplot(x='Pclass',data = titanic_train[titanic_train['Survived']==1],hue='Sex', ax=ax[1]).set_title('P-Class Wise Survive count')
# Above graph is showing Those who travel in 1st Class survived more

titanic_train.columns
# Above graph is showing Those who travel alone survived more

# Create traing and test data set

from sklearn.model_selection import train_test_split

features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']

X = titanic_train[features]

y = titanic_train['Survived'].values



#Normalization of data to give data zero mean and unit variance.

from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)
# Create a LogisticRegression object

from sklearn.linear_model import LogisticRegression

logRegMod = LogisticRegression()
logRegMod.fit(X_train,y_train)
logRegPred = logRegMod.predict(X_test)
# We can check precision,recall,f1-score using classification report

from sklearn.metrics import classification_report

print(classification_report(y_test,logRegPred))
# Create a KNeighborsClassifier object

from sklearn.neighbors import KNeighborsClassifier
# Find the good K value

error_rate = []

for i in range(1,40):

    

    kNeighborMod = KNeighborsClassifier(n_neighbors=i)

    kNeighborMod.fit(X_train,y_train)

    pred_i = kNeighborMod.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# As per above grap we can go with 10 for K value

kNeighborMod = KNeighborsClassifier(n_neighbors=10)

kNeighborMod.fit(X_train,y_train)

kNeighborPred = kNeighborMod.predict(X_test)
# We can check precision,recall,f1-score using classification report

print(classification_report(y_test,kNeighborPred))
# Create an object for DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

dtMod = DecisionTreeClassifier(criterion="entropy")

dtMod.fit(X_train,y_train)

dtPred = dtMod.predict(X_test)
# We can check precision,recall,f1-score using classification report

print(classification_report(y_test,dtPred))
# Create an object for RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
# We can go with 20

rfMod = RandomForestClassifier(n_estimators=20)

rfMod.fit(X_train,y_train)

rfcPred = rfMod.predict(X_test)
# We can check precision,recall,f1-score using classification report

print(classification_report(y_test,rfcPred))
# Create a vector machine and GridSearch object

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
# To determind the value for C and gamma

param_grid = {'C': [0.01,0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
# We can check precision,recall,f1-score using classification report

print(classification_report(y_test,grid_predictions))
# With all model (Logic Regression, K Neighbors Classifier, Decission Tree, Random Forest and SVC ) 

# Random Forect showing more accurate. So we can test our final test dataset with Random Forest

X_test_submit=titanic_test[features]

X_test_submit = preprocessing.StandardScaler().fit(X_test_submit).transform(X_test_submit)

finalPred = rfMod.predict(X_test_submit)
# Read gender file 

df_gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

df_pred = pd.DataFrame({"PassengerId":df_gender['PassengerId'],"Survived":finalPred})
df_pred.head()
sns.countplot(x='Survived',data=df_pred,palette='RdBu_r')
df_pred.to_csv('prediction.csv', encoding='utf-8', index=False)