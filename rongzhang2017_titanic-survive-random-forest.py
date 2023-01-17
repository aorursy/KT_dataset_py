import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import re
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.info()
test.info()
surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]

print(f"Survived:",len(surv), round(len(surv)/len(train)*100.0, 2), 
      "perecnt;\n""Not Survived:",len(nosurv), round(len(nosurv)/len(train)*100.0, 2), "perecnt.\n"
     "Total:", len(train))
plt.subplot()
sns.barplot('Sex', 'Survived', data=train)
plt.subplot()
sns.barplot('Pclass', 'Survived', data=train)
plt.subplot()
sns.barplot('Embarked', 'Survived', data=train)
# Survival by number of sibling/spouse
plt.subplot()
sns.barplot('SibSp', 'Survived', data=train)
# Survival by parch
plt.subplot()
sns.barplot('Parch', 'Survived', data=train)
fig, ax = pyplot.subplots(1,figsize = (12,9))
sns.distplot(train['Age'].dropna(), bins = 16, ax = ax)
print("Missing Ages: ", len(train[train['Age'].isnull()]))
#drop the null Age values
#remove the null data for now
train_drop = train.copy()
train_drop = train_drop.dropna(subset = ['Age'])

#encoding the sex into 0s and 1s
def encodeSex(sex):
    if sex == "male": return 0
    #if female
    return 1

#encoding the ages of passengers into groups
def encodeAge(age):
    return int(age/5)

#encode the data in the DataFrame
train_drop.Sex = train_drop.Sex.apply(encodeSex)
train_drop.Age = train_drop.Age.apply(encodeAge)
#The features that seem important to survival
features = ['Pclass','Sex', 'Age', 'Fare']
X = train_drop[features]
y = train_drop.Survived
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

#create a RandomForestClassifier
forest = RandomForestClassifier(max_leaf_nodes = 55)
forest.fit(train_X, train_y)
print("Feature importance: ", forest.feature_importances_)
print("Accuracy: ", forest.score(test_X, test_y))
#Some of the code here is borrowed

def extractTitle(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        return title.group(1)
    return ""
    
#Create a new column in the data called title
train_with_title = train.copy()
train_with_title = train_with_title.dropna(subset = ['Age'])
train_with_title['Title'] = train_with_title.Name.apply(extractTitle)

#plot the age against the title
fig, ax = pyplot.subplots(1,figsize = (18,6))
sns.scatterplot(x = "Title", y = "Age", data = train_with_title, ax = ax)
#encode the titles
def encodeTitle(title):
    if title == "Master": return 0
    if (title == "Miss" or title == "Mme" or
       title == "Ms" or title == "Mlle"): return 1
    if (title == "Mr" or title == "Mrs" or
       title == "Countess" or title == "Jonkheer"): return 2
    if (title == "Rev" or title == "Dr"): return 3
    if (title == "Don" or title == "Major" or
        title == "Lady" or title == "Sir"): return 4
    return 5

train_with_title.Title = train_with_title.Title.apply(encodeTitle)

#plot the age against the title
fig, ax = pyplot.subplots(1,figsize = (18,6))
sns.scatterplot(x = "Title", y = "Age", data = train_with_title, ax = ax)

test_features = ["Title"]

#We can test a decision tree on train_with_title
line = LinearRegression()

X = train_with_title[test_features]
y = train_with_title.Age

#training the data
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
line.fit(train_X, train_y)
print("Accuracy: ", line.score(test_X, test_y))
test_features = ["Title", "SibSp", "Parch", "Pclass", "Fare"]

X = train_with_title[test_features]
y = train_with_title.Age

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
line.fit(train_X, train_y)
print("Accuracy: ", line.score(test_X, test_y))
predicted_ages = train[train['Age'].isnull()]
predicted_ages['Title'] = predicted_ages.Name.apply(extractTitle)
predicted_ages['Title'] = predicted_ages.Title.apply(encodeTitle)
predictions = pd.Series(line.predict(predicted_ages[test_features]))

train_with_ages = train
#fill in the missing values
#There should be a cleaner and quicker way to write this
#because doing this is really really slow
inc = 0
for i in range(0, len(train['Age'])):
    if math.isnan(train['Age'][i]):
        train_with_ages['Age'][i] = predictions[inc]
        inc += 1
#encode the new df
#encode the data in the DataFrame
train_with_ages.Sex = train_with_ages.Sex.apply(encodeSex)
train_with_ages.Age = train_with_ages.Age.apply(encodeAge)

features = ['Pclass','Sex', 'Age', 'Fare']

X = train_with_ages[features]
y = train_with_ages.Survived

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

#fit the RandomForestClassifier
forest = RandomForestClassifier(max_leaf_nodes = 55)
forest.fit(train_X, train_y)
print("Feature importance: ", forest.feature_importances_)
print("Accuracy: ", forest.score(test_X, test_y))
#making, training and submitting test
predicted_ages = test[test['Age'].isnull()]
predicted_ages['Title'] = predicted_ages.Name.apply(extractTitle)
predicted_ages['Title'] = predicted_ages.Title.apply(encodeTitle)
predictions = pd.Series(line.predict(predicted_ages[test_features]))

#fill in the missing values
#There should be a cleaner and quicker way to write this
inc = 0
for i in range(0, len(test['Age'])):
    if math.isnan(test['Age'][i]):
        test['Age'][i] = predictions[inc]
        inc += 1
        
#encode the new df
#encode the data in the DataFrame
test.Sex = test.Sex.apply(encodeSex)
test.Age = test.Age.apply(encodeAge)
test['Fare'] = test['Fare'].fillna(value = 13)

features = ['Pclass','Sex', 'Age', 'Fare']
X = test[features]

final_predict = forest.predict(X)
prediction = pd.DataFrame(test.PassengerId)
prediction['Survived'] = final_predict.astype('int')

prediction.to_csv('predict.csv',index = False)