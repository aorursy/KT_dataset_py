#Let's get started...

#First, import all the required libraries.

import pandas as pd

import numpy as np

import re                                             #To perform operations on regular expression.

from sklearn.tree import DecisionTreeClassifier       #We'll use a decision tree classifier to predict results.

from sklearn.model_selection import cross_val_score

import warnings                                       #ignoring the warnings generated during some computation

warnings.filterwarnings("ignore")

train_data = pd.read_csv('../input/train.csv')        #loading the train data

test_data = pd.read_csv('../input/test.csv')          #loading the test data

complete_data = [train_data, test_data]               #all data grouped together
print( train_data[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean() )
print( train_data[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean() )
for data in complete_data:

    data['family_size'] = data['SibSp'] + data['Parch'] + 1

print( train_data[["family_size","Survived"]].groupby(["family_size"], as_index = False).mean() )
for data in complete_data:

    data['is_alone'] = 0

    data.loc[data['family_size'] == 1, 'is_alone'] = 1

print (train_data[['is_alone', 'Survived']].groupby(['is_alone'], as_index=False).mean())
for data in complete_data:

    data['Embarked'] = data['Embarked'].fillna('S')

print( train_data[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean() )
for data in complete_data:

    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

train_data['category_fare'] = pd.qcut(train_data['Fare'], 4)

print( train_data[["category_fare","Survived"]].groupby(["category_fare"], as_index = False).mean() )
for data in complete_data:

    age_avg  = data['Age'].mean()

    age_std  = data['Age'].std()

    age_null = data['Age'].isnull().sum()

    #The feature 'Age' has some missing values. We'll fill it with some random values for easy grouping.

    random_list = np.random.randint(age_avg - age_std, age_avg + age_std , size = age_null)

    data['Age'][np.isnan(data['Age'])] = random_list

    data['Age'] = data['Age'].astype(int)



train_data['category_age'] = pd.cut(train_data['Age'], 5)               #grouping data into a set of 5

print( train_data[["category_age","Survived"]].groupby(["category_age"], as_index = False).mean() )
#The feature 'Name' contains different titles of different persons.

#We'll extract the all titles and group them under a list called 'title'

#This will help us to get clean data.

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\. ', name)

    if title_search:

        return title_search.group(1)

    return ""



for data in complete_data:

    data['title'] = data['Name'].apply(get_title)



for data in complete_data:

    data['title'] = data['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')

    data['title'] = data['title'].replace('Mlle','Miss')

    data['title'] = data['title'].replace('Ms','Miss')

    data['title'] = data['title'].replace('Mme','Mrs')

    

print(pd.crosstab(train_data['title'], train_data['Sex']))

print("----------------------")

print(train_data[['title','Survived']].groupby(['title'], as_index = False).mean())
#Mapping the data.

#ML algorithms are efficient with numerical values. 

#We'll convert each of the feature values into a designated numerical value.

#We'll drop some of the unnecessary features which might have an impact on our outcome.

for data in complete_data:



    #Mapping Sex

    sex_map = { 'female':0 , 'male':1 }

    data['Sex'] = data['Sex'].map(sex_map).astype(int)



    #Mapping Title

    title_map = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}

    data['title'] = data['title'].map(title_map)

    data['title'] = data['title'].fillna(0)



    #Mapping Embarked

    embark_map = {'S':0, 'C':1, 'Q':2}

    data['Embarked'] = data['Embarked'].map(embark_map).astype(int)



    #Mapping Fare

    data.loc[ data['Fare'] <= 7.91, 'Fare']                            = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2

    data.loc[ data['Fare'] > 31, 'Fare']                               = 3

    data['Fare'] = data['Fare'].astype(int)



    #Mapping Age

    data.loc[ data['Age'] <= 16, 'Age']                       = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[ data['Age'] > 64, 'Age']                        = 4



#Feature Selection

#Create list of columns to drop

drop_elements = ["Name", "Ticket", "Cabin", "SibSp", "Parch", "family_size"]



#Drop columns from both data sets

train_data = train_data.drop(drop_elements, axis = 1)

train_data = train_data.drop(['PassengerId','category_fare', 'category_age'], axis = 1)

test_data = test_data.drop(drop_elements, axis = 1)



#Print ready to use data

print(train_data.head(10))
#Splitting the data.

#Train set & Test set.

X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()
#Making predictions.

#Using Decision Tree classifier from 'sklearn.tree' 

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

accuracy = round(decision_tree.score(X_train, Y_train) * 100, 2)

print("Model Accuracy: ",accuracy)
#Create a CSV with results

submission = pd.DataFrame({

    "PassengerId": test_data["PassengerId"],

    "Survived": Y_pred

})

submission.to_csv('submission.csv', index = False)