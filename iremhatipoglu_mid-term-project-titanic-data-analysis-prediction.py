import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt #matplotlib and seaborn are for the graphics that we are going to use

import seaborn as sns

import xgboost as xgb #for modelling

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

#We added useful libraries
train = pd.read_csv("/kaggle/input/titanic/train.csv") #We loaded dataset

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassengerId = test['PassengerId']

full = pd.concat([train, test], keys=['train','test'])

train.head(10) #We used head method for to see first 10 rows of our data

train.count() #We used count method for to learn how many people were on the Titanic
train.describe() #Descriptive statistics table of the data set
train.info() #We used for to see data types of columns
sns.factorplot('Sex',data=train,kind='count') #this graphic shows how many men and women on the ship

print("Number of female: ", len(train.groupby('Sex').groups['female'])) #we used groupby function for to find the number of women

print("Number of male: ", len(train.groupby('Sex').groups['male'])) #we used groupby function for to find the number of men
sns.countplot(x='Survived', hue='Sex', data=train) #Gender based survivors
#Number of passengers according to class

train['Pclass'].value_counts()
train['Pclass'].value_counts().plot(kind='barh', color='coral', figsize=[16,4])

plt.xlabel('Frequency')

plt.ylabel('Pclass')

plt.show()
sns.countplot(x='Pclass', hue='Sex', data=train) 
train[train["Name"].str.contains("Brown")] #we seached names contains "Brown" as her lastname.
sns.countplot(x='Survived', hue='Pclass', data=train) #surviving numbers for class of travel
#We diversified age groups by defining a function

def age_dis(x):

    if x>=0 and x <12: #we accepted that the age under 12 years old are child

        return 'Child'

    elif x>=12 and x<=20:

        return 'Young'

    else:

        return 'Adult' #we accepted that the age above 20 years old are adult
train['Age'].apply(age_dis).value_counts() #age based numbers
#Visualization of percentages of passengers by age

train['Age'].apply(age_dis).value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.title('Distribution of passengers by age')

plt.show()
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train[train['Sex']=='female']

men = train[train['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass',y='Age',data=train)
#We are going to spot some more features, that contain missing values (NaN = not a number)



total = train.isnull().sum().sort_values(ascending=False)

percent_1 = train.isnull().sum()/train.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
train["Embarked"] = train["Embarked"].fillna('S') #filled with S
sns.countplot(x='Survived', hue='Embarked', data=train) #Surviving rates based on embarking spots
def add_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        return int(train[train["Pclass"] == Pclass]["Age"].mean()) #We obtained the average age with .mean () function

    else:

        return Age
train["Age"] = train[["Age", "Pclass"]].apply(add_age,axis=1) #we call the function
train.drop("Cabin",inplace=True,axis=1) #we removed Cabin with .drop() function
train.isnull().sum() #we removed rows with null values
name = train['Name']

train['Title'] = [i.split(".")[0].split(",")[-1].strip() for i in name] #we split and create a new feature
train['Title'].head(10) #to see first 10 rows of our feature
sns.countplot(x='Title', data=train) #to show number of titles

plt.xticks(rotation= 90)

plt.show()
data = [train]

titles = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with Rare

    dataset['Title'] = dataset['Title'].replace(['Don', 'Rev','Dr', 'Major',\

                                            'Lady', 'Sir', 'Col', 'Capt','the Countess','Jonkheer'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train = train.drop(['Name'], axis=1)

sns.countplot(x="Title", data = train)

plt.show()

# 0 : Mr

# 1 : Mrs

# 2 : Miss

# 3 : Master

# 4 : Rare
train = pd.get_dummies(train,columns=["Title"])

train.head()
data = [train]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train]



for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)
train["Sex"] = train["Sex"].astype("category")

train = pd.get_dummies(train, columns=["Sex"])

train.head()
train['Ticket'].describe()
train= train.drop(["Ticket", "PassengerId"], axis=1) # I also dropped Passenger Id  too cause it's unnecessary.
ports = {"S": 0, "C": 1, "Q": 2}

data = [train]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
train = pd.get_dummies(train, columns=["Embarked"])

train.head()
train['Pclass'] = train['Pclass'].astype("category")

train = pd.get_dummies(train, columns= ['Pclass'])

train.head()
#importing necessary libraries

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from scipy.stats.stats import pearsonr

from xgboost import XGBClassifier

from sklearn.preprocessing import MinMaxScaler
X = train.drop("Survived",axis=1) #x will contain all the features and y will contain the target variable

y = train["Survived"]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)



print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))

# Making a list of accuracies

accuracies = []
rdmf = RandomForestClassifier(n_estimators=20, criterion='entropy')

rdmf.fit(X_train, y_train)
#writing the accuracy score

rdmf_score = rdmf.score(X_test, y_test)

rdmf_score_tr = rdmf.score(X_train, y_train)

accuracies.append(rdmf_score)

print(rdmf_score)

print(rdmf_score_tr)
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
#writing the accuracy score

lr_score = classifier.score(X_test, y_test)

accuracies.append(lr_score)

print(lr_score)
knn = KNeighborsClassifier(p=2, n_neighbors=10)

knn.fit(X_train, y_train)
#writing the accuracy score

knn_score = knn.score(X_test, y_test)

accuracies.append(knn_score)

print(knn_score)
svm = SVC(kernel='linear')

svm.fit(X_train, y_train)
#writing the accuracy score

svm_score = svm.score(X_test, y_test)

accuracies.append(svm_score)

print(svm_score)
k_svm = SVC(kernel='rbf')

k_svm.fit(X_train, y_train)
#writing the accuracy score

k_svm_score = k_svm.score(X_test, y_test)

accuracies.append(k_svm_score)

print(k_svm_score)
xgb = XGBClassifier()

xgb.fit(X_train, y_train)
#writing the accuracy score

xgb_score = xgb.score(X_test, y_test)

accuracies.append(xgb_score)

print(xgb_score)
accuracy_labels = ['Random Forest', 'Logistic Regression', 'KNN', 'LSVM', 'Kernel SVM', 'Xgboost']
accuracy_frq= sns.barplot(x=accuracies, y=accuracy_labels)
predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

#confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
#prediction and submission

test_survived = pd.Series(classifier.predict(X_test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic_output.csv", index = False) #output