

import numpy as np #linear algebra

import pandas as pd # data processing, CSV file 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
print(round(100/train_data.count().PassengerId*train_data[(train_data.Survived==1) & (train_data.Sex=='male')].count().PassengerId,2), '% chance for at mænd dør')
train_data['Ticket'].isnull().value_counts() #Check for missing values
train_data.info() 
train_data['Ticket_cat']=train_data['Ticket'].astype('category')

train_data['Ticket_cat_codes']=train_data['Ticket_cat'].cat.codes

train_data[['Ticket','Ticket_cat','Ticket_cat_codes']]
test_data['Ticket_cat']=test_data['Ticket'].astype('category')

test_data['Ticket_cat_codes']=test_data['Ticket_cat'].cat.codes

test_data[['Ticket','Ticket_cat','Ticket_cat_codes']]
train_data['Sex_cat'] = train_data['Sex'].astype("category")

train_data['Sex_cat_codes'] = train_data['Sex_cat'].cat.codes

train_data[['Sex','Sex_cat','Sex_cat_codes']]

test_data['Sex_cat'] = test_data['Sex'].astype("category")

test_data['Sex_cat_codes'] = test_data['Sex_cat'].cat.codes

test_data[['Sex','Sex_cat','Sex_cat_codes']]
train_data['Age'].isnull().value_counts() 
train_data['Age'].isnull().values.any() #to check if there is any NaN's in the Ticket column

train_data['Age_filled']=train_data.Age.fillna(train_data.Age.mean()) #Fill the NaN's with the mean value

train_data.Age_filled

test_data['Age_filled']=test_data.Age.fillna(test_data.Age.mean()) #Fill the NaN's with the mean value

test_data.Age_filled

train_data['Age_filled_cat'] = train_data['Age_filled'].astype("category")

train_data['Age_filled_cat_codes'] = train_data['Age_filled_cat'].cat.codes

train_data[['Age_filled','Age_filled_cat','Age_filled_cat_codes']]

test_data['Age_filled_cat'] = test_data['Age_filled'].astype("category")

test_data['Age_filled_cat_codes'] = test_data['Age_filled_cat'].cat.codes

test_data[['Age_filled','Age_filled_cat','Age_filled_cat_codes']]
train_data['Fare'].isnull().values.any() #to check if there is any NaN's in the Fare column

train_data['Fare_cat'] = train_data['Fare'].astype("category")

train_data['Fare_cat_codes'] = train_data['Fare_cat'].cat.codes

train_data[['Fare','Fare_cat','Fare_cat_codes']]

test_data['Fare_cat'] = test_data['Fare'].astype("category")

test_data['Fare_cat_codes'] = test_data['Fare_cat'].cat.codes

test_data[['Fare','Fare_cat','Fare_cat_codes']]
train_data['Embarked_filled']=pd.Categorical(train_data['Embarked'])

def setCategory(train_data):

    if train_data['Embarked_filled'] == 'S':

        return 'Southampton'

    elif train_data['Embarked_filled'] == 'C':

        return 'Cherbourg'

    elif train_data['Embarked_filled'] == 'Q':

        return 'Queenstown'

train_data['Embarked_filled'] = train_data.apply(setCategory, axis =1) #Replacing letters with names

test_data['Embarked_filled']=pd.Categorical(test_data['Embarked'])

def setCategory(test_data):

    if test_data['Embarked_filled'] == 'S':

        return 'Southampton'

    elif test_data['Embarked_filled'] == 'C':

        return 'Cherbourg'

    elif test_data['Embarked_filled'] == 'Q':

        return 'Queenstown'

test_data['Embarked_filled'] = test_data.apply(setCategory, axis =1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch","Fare_cat_codes","Embarked_filled"]

features1= ["Sex","Ticket_cat_codes", "Age_filled_cat_codes"]



X = pd.get_dummies(train_data[features])  #Using RandomForestClassifier

X1 = pd.get_dummies(train_data[features1])

X_test = pd.get_dummies(test_data[features])

X1_test = pd.get_dummies(test_data[features1])



clf = LogisticRegression(random_state=123) #Using LogisticRegression

clf.fit(X1,y)

y_pred=clf.predict(X1_test)

clf.score(X1,y)



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

output