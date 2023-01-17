import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
def check_type(df):

    '''Function to check the data types of each attribute in a dataframe

    accepts a dataframe as an input and prints the name of the attribute and its data type'''

    for name, dtype in df.dtypes.iteritems():

        print(name, dtype)
display(check_type(test))

display(check_type(train))
train.corr() #Check correlation between each attribute
#Some insights about Survival rates based on their embarkment and sex



C = train.loc[train.Embarked == 'C']['Survived']

rate_embarked_C = round((sum(C) / len(C))*100,2)



print("% of people boarded at Cherbourg survived is", rate_embarked_C)



S = train.loc[train.Embarked == 'S']['Survived']

rate_embarked_S = round((sum(S) / len(S))*100, 2)



print("% of people boarded at Southampton survived is", rate_embarked_S)



Q = train.loc[train.Embarked == 'Q']['Survived']

rate_embarked_Q = round((sum(Q) / len(Q))*100,2)

print("% of people boarded at Queenstown survived is", rate_embarked_Q)



M = train.loc[train.Sex == 'male']['Survived']

F = train.loc[train.Sex == 'female']['Survived']

male_rate = round((sum(M) / len(M))*100,2)

female_rate = round((sum(F) / len(F))*100,2)



print("Male survivor rate is", male_rate)

print("Female survivor rate is", female_rate)



#Passengers boarded at Cherbourg has maximum survival rate compared to others 

#About 74% of female passengers survived overall
train_df = train[['Pclass', 'Sex', 'Embarked']].copy()

test_df = test[['Pclass', 'Sex', 'Embarked']].copy()

display(train_df.isna().sum())#check NAs

display(test_df.isna().sum())



#Fill NAs

display(train_df['Embarked'].describe())

common_value = 'S'

dataset = [train_df, test_df]

genders = {'male': 0, 'female':1}

ports = {'C' :0, 'Q': 1, 'S': 2}

for data in dataset:

    data['Sex'] = data['Sex'].map(genders)

    data['Embarked'] = data['Embarked'].fillna(common_value)

    data['Embarked'] = data['Embarked'].map(ports)

#Confirm if NAs are present in the dataset

display(train_df.isna().sum())

display(test_df.isna().sum())
#Convert the dataframe to int for ML models to process

train_df = train_df.astype(int)

test_df = test_df.astype(int)
features = ['Pclass','Sex','Embarked']

x_train = train_df[['Pclass','Sex','Embarked']]

y = train[['Survived']]

x_test = test_df[['Pclass','Sex','Embarked']]
from sklearn.tree import DecisionTreeClassifier as DTree



classifier = DTree(random_state = 0, splitter = 'random')

classifier.fit(train_df, y)

predictions = classifier.predict(x_test)

accuracy = round(classifier.score(x_train, y)*100,2)

display(accuracy)
from sklearn.ensemble import RandomForestClassifier as RFC

model = RFC(random_state = 1)

model.fit(train_df, y)

pred = model.predict(x_test)

accuracy = round(classifier.score(x_train, y)*100,2)

display(accuracy)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier, x_train, y, cv=10, scoring = 'accuracy')

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
scores = cross_val_score(model, x_train, y, cv=10, scoring = 'accuracy')

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output

output.to_csv('DecisionTree.csv', index=False)

print("Your submission was successfully saved!")
