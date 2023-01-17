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
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.shape
train.info()
test.shape
test.head()
train.isnull().sum()



train.drop(columns=['Cabin'],inplace=True,axis=1)

test.drop(columns=['Cabin'],inplace=True,axis=1)
#dealing with missing values Fare(test)

#age

#embarked(train)



test['Fare']=test['Fare'].fillna(test['Fare'].mean())
test.isnull().sum()
train['Embarked'].value_counts()

train['Embarked']=train['Embarked'].fillna('S')
train.isnull().sum()
age_train=np.random.randint(train['Age'].mean()-train['Age'].std(),train['Age'].mean()+train['Age'].std(),177)

age_test=np.random.randint(test['Age'].mean()-test['Age'].std(),test['Age'].mean()+test['Age'].std(),86)
train['Age'][train['Age'].isnull()]=age_train
test['Age'][test['Age'].isnull()]=age_test
test.isnull().sum()
train['family']=train['SibSp']+train['Parch']+1

test['family']=test['SibSp']+test['Parch']+1
train['family_size']=0

test['family_size']=0





def family_size(no):

    if no==1:

        return "alone"

    elif no>1 and no<=4:

        return "small"

    else:

        return "large"
#very important

train['family_size']=train['family'].apply(family_size)

test['family_size']=test['family'].apply(family_size)
train.sample(3)
test.sample(3)
for df in [train, test]:

    for row in df.index:

        # take the text after ", " and before ". "

        df.loc[row, 'Title'] = df.loc[row, 'Name'].split(", ")[1].split(". ")[0]







train.drop(columns=['SibSp','Parch','family','Ticket','PassengerId'],inplace=True,axis=1)

#passenger_id=test['PassengerId']

test.drop(columns=['SibSp','Parch','family','Ticket'],inplace=True,axis=1)
train.head()
temp2_df = pd.concat([train.copy(), test.copy()], sort = False)



print("Number of unique titles (in both dataframes):", temp2_df.Title.nunique()) # shows the number of unique values

print("Number of null titles (in both dataframes):", temp2_df.Title.isna().sum()) # shows the number of null values



# then we count the rows (by arbitrarily counting the 'Name' values)

temp2_df.groupby("Title").count().Name



prof_titles = ["Capt", "Col", "Dr", "Major", "Rev"] # military or professional titles

other_titles = ["Don", "Dona", "Jonkheer", "Lady", "Mlle", "Mme", "Ms", "Sir", "the Countess"] # nobility/foreign titles

for df in [train, test]:

    for row in df.index:

        if df.loc[row, 'Title'] in prof_titles:

            df.loc[row, 'Title'] = "Professional"

        elif df.loc[row, 'Title'] in other_titles:

            df.loc[row, 'Title'] = "Other"

            

            

train.groupby("Title").Survived.agg([("Count", "count"), ("Survival (mean)", "mean")], axis = "rows")
for df in [train,test]:

    df.drop(columns = ['Name'], inplace = True)
temp3_df = pd.concat([train.copy(), test.copy()], sort = False)

title_ages = temp3_df.groupby("Title").Age.mean()

title_ages
for df in [train,test]:

    for row in df.index:

        if pd.isna(df.loc[row, 'Age']): # if 'Age' is null value ("NaN")

            df.loc[row, 'Age'] = title_ages[df.loc[row, 'Title']] # then set to average for that passenger's title, as above
from sklearn.preprocessing import LabelEncoder

LR=LabelEncoder()



train['Pclass']=LR.fit_transform(train['Pclass'])

test['Pclass']=LR.transform(test['Pclass'])

train['Sex']=LR.fit_transform(train['Sex'])

test['Sex']=LR.transform(test['Sex'])

train['family_size']=LR.fit_transform(train['family_size'])

test['family_size']=LR.transform(test['family_size'])

train['Embarked']=LR.fit_transform(train['Embarked'])

test['Embarked']=LR.transform(test['Embarked'])



test.head()
train=pd.get_dummies(train,columns=['Pclass','Embarked','family_size'],drop_first=True)

test=pd.get_dummies(test,columns=['Pclass','Embarked','family_size'],drop_first=True)

train.sample(6)
test.sample(5)

train.drop(columns=['Title'],inplace=True,axis=1)

test.drop(columns=['Title'],inplace=True,axis=1)



X=train.iloc[:,1:].values

y=train.iloc[:,0].values



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.20,random_state=1)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=1000,random_state =0, max_depth = 7,max_features=0.4,min_samples_leaf=3)

clf.fit(X_train, y_train)



print("Train Accuracy:",clf.score(X_train, y_train))

print("Test Accuracy:", clf.score(X_test, y_test))

validate=test.drop('PassengerId',axis=1)

prediction = clf.predict(validate)

test
prediction
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': prediction})

output.to_csv('my_submission.csv', index = False)