#libraries

import pandas as pd

import numpy as np

import re

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
#Training (891 Entries) & Testing (418 Entries) data

train_data = pd.read_csv('../input/test-dataset-for-titanic-competition/titanic_train.csv')

test_data = pd.read_csv('../input/test-dataset-for-titanic-competition/titanic_test.csv')

all_data = [train_data, test_data]
#to know the rows and column of a train_data

train_data.shape
#Training (891 Entries)

train_data.info()
#To get top 5 enteries of train_data

train_data.head()
#to see how many null value in Train_data set

train_data.isnull().sum()
#to know the rows and column of a test_data

test_data.shape
#Testing (418 Entries)

test_data.info()
test_data.head()
#To know the number of null values in each column

test_data.isnull().sum()
#Bar chat function

def bar_chart(feature):

    survived = train_data[train_data['Survived']==1][feature].value_counts()

    dead= train_data[train_data['Survived']==0][feature].value_counts()

    df=pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True,figsize=(10,5))
#Feature 1: Pclass

print( train_data[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean() )

bar_chart('Pclass')
#Feature 2: Sex

print( train_data[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean() )

bar_chart('Sex')
#Feature 3: Family

for data in all_data:

    data['family_size']=data['SibSp']+data['Parch']+1



#print(train_data[["family_size","Survived"]].groupby(["family_size"],as_index=False).mean())

bar_chart('family_size')
#Feature 3.1: is alone?



for data in all_data:

    data['is_alone']=0

    data.loc[data['family_size']==1,'is_alone']=1



#print(train_data[['is_alone','Survived']].groupby(['is_alone'],as_index=False).mean())

bar_chart('is_alone')
#Feature 4: Embarked part 1





Pclass1 = train_data[train_data['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train_data[train_data['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train_data[train_data['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1,Pclass2,Pclass3])

df.index = ['lst class','2nd class','3rd class']



#print(train_data[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean())

df.plot(kind='bar',stacked=True,figsize=(10,5))
#Feature 4: Embarked part 2

for data in all_data:

    data['Embarked']=data['Embarked'].fillna('S')

    
#Feature 5: Fare

for data in all_data:

    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    

train_data['category_fare']=pd.qcut(train_data['Fare'],4)



print(train_data[["category_fare","Survived"]].groupby(["category_fare"],as_index=False).mean())

bar_chart('category_fare')
#Feature 6: Name part 1

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\. ', name)

    if title_search:

        return title_search.group(1)

    return ""



for data in all_data:

    data['title'] = data['Name'].apply(get_title)



data['title'].value_counts()
#Feature 6: Name part 2



#replacing every title with the common title 

for data in all_data:

    data['title'] = data['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')

    data['title'] = data['title'].replace('Mlle','Miss')

    data['title'] = data['title'].replace('Ms','Miss')

    data['title'] = data['title'].replace('Mme','Mrs')

    

#We compute the name title with Sex.

print(pd.crosstab(train_data['title'], train_data['Sex']))

print("----------------------")



print(train_data[['title','Survived']].groupby(['title'], as_index = False).mean())

bar_chart('title')
#Feature 7: Age

#train_data['Age'].fillna(train_data.groupby("title")["Age"].transform("median"), inplace=True)

for data in all_data:

    age_avg  = data['Age'].mean()

    age_std  = data['Age'].std()

    age_null = data['Age'].isnull().sum()



    random_list = np.random.randint(age_avg - age_std, age_avg + age_std , size = age_null)

    data['Age'][np.isnan(data['Age'])] = random_list

    data['Age'] = data['Age'].astype(int)



train_data['category_age'] = pd.cut(train_data['Age'], 5)

print( train_data[["category_age","Survived"]].groupby(["category_age"], as_index = False).mean() )

bar_chart('category_age')
#Map Data

for data in all_data:



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

drop_elements = ["Name", "Ticket", "Cabin", "SibSp", "Parch"]



#Drop columns from both data sets

train_data = train_data.drop(drop_elements, axis = 1)

train_data = train_data.drop(['PassengerId','category_fare', 'category_age'], axis = 1)

test_data = test_data.drop(drop_elements, axis = 1)



#Print ready to use data

print(train_data.head(10))
#Prediction

#Train and Test data

X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()
#Running our classifier

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