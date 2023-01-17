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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.drop(columns=['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],inplace=True)

train.head()
test.drop(columns=['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],inplace=True)

test.head()
from sklearn.preprocessing import LabelEncoder 

  

le = LabelEncoder() 

for df in [train,test]:  

    df['Sex_binary'] = le.fit_transform(df['Sex']) 

df['Sex_binary'].head()
train["Age"] = train["Age"].fillna(0)

test["Age"] = test["Age"].fillna(0)
train["Age"].isnull().value_counts()

train.drop(columns=["Sex"])

test.drop(columns=["Sex"])
features = ['Pclass','Age','Sex_binary']

target = 'Survived'
train[features].head(3)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(train[features],train[target])
predictions = classifier.predict(test[features])

predictions

print(classifier.score(train[features],train[target]))
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})



#Visualize the first 5 rows

submission.head()


#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions 3.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)