#importing all needed modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

#setting visualization style and inline plots
%matplotlib inline
sns.set()
#importing the dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# taking a look at the first 5 rows of the dataset
train.head(n=5)
#taking a look at the overall data
train.info()
#summary statistics of numeric training data
train.describe()
#plotting how many people survived from the training dataset
sns.countplot(x='Survived', data=train)
# creating a base model(bad model) that predicts that nobody survived
test['Survived'] = 0
test[['PassengerId','Survived']].to_csv('no_survived.csv', index=False)
#Now that we have a benchmark, let's look at how features influence the output (finding relationships)
#looking at features of the dataset based on survival
sns.factorplot(x='Survived', col='Sex', kind='count', data=train)
#We see that females survived more than males
#To get numbers
print(train.groupby(['Sex']).Survived.sum())
#calculating proportions of survivors
print("\n Proportions: ")
print(train[train.Sex=='female'].Survived.sum()/train[train.Sex == 'female'].Survived.count())
print(train[train.Sex=='male'].Survived.sum()/train[train.Sex == 'male'].Survived.count())
#predicting that all females survived and all males did not - still a BAD MODEL! :P
test['Survived'] = test.Sex =='female'
test['Survived'] = test.Survived.apply(lambda x: int(x))
test.head()
#printing this out to a csv
test[['PassengerId','Survived']].to_csv('no_survived.csv', index=False)
#Doing some exploratory data analysis now
sns.factorplot(x='Survived', col='Pclass',kind='count', data=train)
sns.factorplot(x='Survived', col='Embarked', kind='count',data=train)
#plotting fare on histogram
sns.distplot(train.Fare,kde=False)
#we can see that a lot of people paid less than 100 to board the titanic
#plotting survival on fare
train.groupby('Survived').Fare.hist(alpha=0.6)
#Let us look at age now
#removing null/missing value rows from train
train_drop = train.dropna()
sns.distplot(train_drop.Age,kde=False)
train.groupby('Survived').Age.hist(alpha=0.5)
sns.swarmplot(x='Survived', y='Fare', data=train)
#Looking at Fare and Survival numbers
train.groupby('Survived').Fare.describe()
#Looking at all the features using pairplot
sns.pairplot(train_drop,hue='Survived')
#importing some new libraries
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#removing the survived col into a separate var
survived = train.Survived
#combining train and test datasets
data = pd.concat([train.drop(['Survived'], axis=1),test])
data.info()
#Estimating missing data using median
data['Age'] = data.Age.fillna(data.Age.median())
#note that there is only one missing vlaue for 'Fare'
data['Fare'] = data.Fare.fillna(data.Fare.median())

#check this by checking info again
data.info()
#converting the Sex column to numeric encoding
data = pd.get_dummies(data, columns = ['Sex'], drop_first=True)
data.head()
data = data[['Sex_male','Fare','Age','Pclass','SibSp']]
data.head()
data.info()
#spliiting the data back into train and test sets
data_train = data[:891]
data_test = data[891:]
#converting dataframes to numpy arrays for sklearn
X = data_train.values
test_set = data_test.values
y = survived.values
#build decision-tree classifier
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X,y)
#predicting values for test_set
Y_pred = clf.predict(test_set)
#test.head()
test['Survived'] = Y_pred

#print this to csv
test[['PassengerId','Survived']].to_csv('no_survived.csv', index=False)
#checking for which depth gives max accuracy
#splitting train data intoe train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
#running a loop through depth =0 to 9 and checking accuracy for each
depth = np.arange(1,9)
train_acc = np.empty(len(depth))
test_acc = np.empty(len(depth))

for i,k in enumerate(depth):
    #create decision tree with depth=k
    clf = tree.DecisionTreeClassifier(max_depth=k)
    
    # Fit to training data
    clf.fit(X_train, y_train)

    #Calculate the accuracy on the training set
    train_acc[i] = clf.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_acc[i] = clf.score(X_test, y_test)

plt.plot(depth,train_acc,label='Training accuracy')
plt.plot(depth, test_acc, label= 'Testing accuracy')
plt.legend()
plt.xlabel('Depth')
plt.ylabel('Accuracy')

#Let's try using the other features to predict
#removing the survived col into a separate var
survived = train.Survived
#combining train and test datasets
data = pd.concat([train.drop(['Survived'], axis=1),test])
#Estimating missing data using median
data['Age'] = data.Age.fillna(data.Age.median())
#note that there is only one missing vlaue for 'Fare'
data['Fare'] = data.Fare.fillna(data.Fare.median())
#converting the Sex column to numeric encoding
data = pd.get_dummies(data, columns = ['Sex'], drop_first=True)
data.head()
data.tail()
#We can see that the names have various titles - this could prossibly be related to survival - based on class in society
#extracting titles from names
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.',x).group(1))
sns.countplot('Title', data=data)
plt.xticks(rotation=45)
data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr','Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);
data.tail()
#Since cabin has a large number of missing values - we can assume that maybe a large number of people did not have cabins at all
#creating a new column that encodes whether the passenger had a cabin or not
data['hasCabin'] = ~data.Cabin.isnull()
data.head()# Drop columns and view head
# Dropping unwanted columns
data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data.head()
data.info()
#estimating missing embarked values
print("Southampton: ")
print(data[data['Embarked'] == 'S'].shape[0])

print("Cherbourg: ")
print(data[data['Embarked'] == 'C'].shape[0])

print("Qweenstown: ")
print(data[data['Embarked'] == 'Q'].shape[0])
#We can see that maximum passengers boarded at Southampton, so we can fill the missing vlaues with 'S'
data = data.fillna({'Embarked':'S'})
data.info()
#discretizing the age and fare features into bins
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False )
data['CatFare']= pd.qcut(data.Fare, q=4, labels=False)
data.head()
#Now we can drop age and fare
data = data.drop(['Age', 'Fare'], axis=1)
data.head()
#Calculating the size of families into a new feature
data['Fam_size'] = data.Parch + data.SibSp
data.head()
data.groupby('Survived').Fam_size.hist(alpha=0.5)
#converting into binary features
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()
# Split into test.train
data_train = data_dum.iloc[:891]
data_train = data_train.drop(['Survived'], axis=1)
data_test = data_dum.iloc[891:]
data_test = data_test.drop(['Survived'], axis=1)

# Transform into arrays for scikit-learn
X = data_train.values
test_set = data_test.values
y = survived.values
# Setup the hyperparameter grid
dep = np.arange(1,9)
param_grid = {'max_depth' : dep}

# Instantiate a decision tree classifier: clf
clf = tree.DecisionTreeClassifier()

# Instantiate the GridSearchCV object: clf_cv
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit it to the data
clf_cv.fit(X, y)

# Print the tuned parameter and score
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))
Y_pred = clf_cv.predict(test_set)
test['Survived'] = Y_pred
test[['PassengerId','Survived']].to_csv('no_survived.csv', index=False)
#trying support vector mahcines
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(X, y)
y_pred = linear_svc.predict(test_set)

test['Survived'] = y_pred
test[['PassengerId','Survived']].to_csv('no_survived.csv', index=False)
from sklearn.ensemble import RandomForestClassifier

#builf RF model
clf = RandomForestClassifier(n_jobs = 2, random_state = 1)
clf.fit(X,y)

#computing accuracy on train set
print("Accuracy on train set: ",clf.score(X,y))

#predicting values from test set
Y_pred = clf.predict(test_set)
test['Survived'] = Y_pred
test[['PassengerId','Survived']].to_csv('no_survived.csv', index=False)
#putting values into an excel sheet
test['Survived'] = Y_pred
test[['PassengerId','Survived']].to_csv('no_survived.csv', index=False)