#Getting all the packages we need: 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #statist graph package

import matplotlib.pyplot as plt #plot package



#plt.style.use('ggplot') #choosing favorite R ggplot stype

plt.style.use('bmh') #setting up 'bmh' as "Bayesian Methods for Hackers" style sheet





#loading ML packages:

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



#input files directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read train data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
print("train shape:",train.shape)

print("test shape :",test.shape)
test.head(5)
train.sample(5)
#The mean of the target column:

round(np.mean(train['Survived']), 2)
train['Survived'].value_counts()

#Survival rate:



color = ('#F5736B', '#C7F35B')

plt.pie(train["Survived"].value_counts(), data = train, explode=[0.08,0], labels=("Not Survived", "Survived"), 

        autopct="%1.1f%%", colors=color, shadow=True, startangle=400, radius=1.6, textprops = {"fontsize":20})

plt.show();
train.describe()
h_labels = [x.replace('_', ' ').title() for x in 

            list(train.select_dtypes(include=['number', 'bool']).columns.values)]



fig, ax = plt.subplots(figsize=(10,6))

_ = sns.heatmap(train.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
#How many people from which social class were on Titanic?



sns.countplot(train['Pclass'])
#Let's group social class and survived data:



sns.countplot(x = 'Survived', hue = 'Pclass', data = train)



#0 - didn't survived

#1 - survived
plt.figure(figsize = (16, 8))



sns.distplot(train["Fare"])

plt.title("Fare Histogram")

plt.xlabel("Fare")

plt.show()
#Let's see the same survival chart for gender data: 



sns.countplot(x = 'Survived', hue = 'Sex', data = train)



#0 - didn't survived

#1 - survived
#Age distribution

plt.figure(figsize = (16, 8))



sns.distplot(train["Age"])

plt.title("Age Histogram")

plt.xlabel("Age")

plt.show()
#Let's group age and survived data:

plt.figure(figsize = (35, 8))



sns.countplot(x = 'Age', hue = 'Survived', data = train)



plt.title("Age Histogram")

plt.xlabel("Age")

plt.show()



#0 - didn't survived

#1 - survived

g = sns.FacetGrid(train, col = "Survived")

g.map(sns.distplot, "Age")

plt.show()
sns.countplot(train['SibSp'])

#Let's group family and survived data:

plt.figure(figsize = (15, 8))



sns.countplot(x = 'SibSp', hue = 'Survived', data = train)



plt.title("Siblings/Spouse Histogram")

plt.xlabel("Siblings/Spouse")

plt.show()



#0 - didn't survived

#1 - survived

#Let's group children/parents and survived data:

plt.figure(figsize = (12, 6))



sns.countplot(x = 'Parch', hue = 'Survived', data = train)



plt.title("Parents/Children Histogram")

plt.xlabel("Parents/Children")

plt.show()



#0 - didn't survived

#1 - survived
#Let's review the data types we have to work with:



test.info()
# Let's look for missing values

train.isnull().sum().sort_values(ascending = False)
# And missing values for Test set:

test.isnull().sum().sort_values(ascending = False)
#We have to fill in missing age values in our dataset. We can use Median Titanic passenger age data for this, which is 28 (as we confirmed it above in EDA)



train['Age']=train['Age'].fillna('28')

test['Age']=train['Age'].fillna('28')
#Using the same logic again and filling in "S" for missing Embarked values:



train['Embarked'] = train['Embarked'].fillna('S')

test['Embarked'] = train['Embarked'].fillna('S')
# Convert 'Embarked' variable to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2
#Our test set has one empty value, which we will fill in with the median:  

test['Fare']=train['Fare'].fillna('14')
#Convert categorical Gender column to numerical data:



train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1



test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

#Let's validate we don't have empty values left: 



train.isnull().sum().sort_values(ascending = False)
#And, for the test set:



test.isnull().sum().sort_values(ascending = False)
#Dropping columns which we won't use:



train.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

test.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)
#Final check for the Train data:



train.head(5)
#And, for the test set:



test.head(5)
#Running model for only 20% our test sample using test split feature:



x_train, x_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 

                                                    train['Survived'], test_size = 0.2, 

                                                    random_state = 2)
#Logistic Regression model:



logisticRegression = LogisticRegression(max_iter = 10000)

logisticRegression.fit(x_train, y_train)



# Predicting the values for Survived:

predictions = logisticRegression.predict(x_test)



#print(predictions)



acc_logreg = round(accuracy_score(predictions, y_test) * 100, 2)

print(acc_logreg)
#Decision Tree Classifier:



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)



# Predicting the values for Survived:

predictions = decisiontree.predict(x_test)



#print(predictions)



acc_decisiontree = round(accuracy_score(predictions, y_test) * 100, 5)

print(acc_decisiontree)
#Gaussian NB:



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)



# Predicting the values for Survived:

predictions = gaussian.predict(x_test)



#print(predictions)



acc_gaussian = round(accuracy_score(predictions, y_test) * 100, 5)

print(acc_gaussian)
#Support Vector Machines



svc = SVC(max_iter = 10000)

svc.fit(x_train, y_train)



# Predicting the values for Survived:

predictions = svc.predict(x_test)



#print(predictions)



acc_svc = round(accuracy_score(predictions, y_test) * 100, 2)

print(acc_svc)
# Linear SVC



linear_svc = LinearSVC(max_iter = 10000)

linear_svc.fit(x_train, y_train)



# Predicting the values for Survived:

predictions = linear_svc.predict(x_test)



#print(predictions)



acc_linear_svc = round(accuracy_score(predictions, y_test) * 100, 2)

print(acc_linear_svc)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'Logistic Regression', 'Naive Bayes', 'Linear SVC', 'Decision Tree'],

    'Score': [acc_svc, acc_logreg, acc_gaussian, acc_linear_svc, acc_decisiontree]})

models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 



ids = test['PassengerId']

print(len(ids))

predictions = logisticRegression.predict(test)
#set the output file:

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })



output.tail(5)
output.to_csv('kaggle_titanic_submission.csv', index=False)

print("Successfull submission")