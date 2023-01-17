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
import matplotlib.pyplot as plt #importing the required libraries
import seaborn as  sns
from sklearn.model_selection import cross_val_score
train = pd.read_csv('/kaggle/input/titanic/train.csv') #our training data
test = pd.read_csv('/kaggle/input/titanic/test.csv') #our testing data
train.head() #checking head of our training data
test.head() #checking head of our testing data
train.info() #checking the column names and types in our training data
test.info() #checking name of columns and the type of data in our test set
train['Survived'].value_counts(normalize = True) #counting the number of people that survived
train.isnull().sum() #checking the null values in our data(train)
test.isnull().sum() #checking null values in our data(test)
sns.heatmap(train.isnull()) #plotting the null values
sns.countplot(train['Survived'],data = train).set_title('Survivor counts') #visualizing the survival count
sns.countplot(train['Survived'],data = train,hue = 'Sex') #checking the amount of males and females in our survival set
sns.barplot(x='Sex',y='Survived',data = train) #plotting to find out which gender survived the most
train.groupby('Pclass').Survived.mean() #taking the mean of the ssurvived data according to the PClass
sns.barplot(x='Pclass',y='Survived',data = train) #checking the amount of passengers that survived on basis of PClass
sns.countplot(x='Pclass',hue = 'Survived',data = train)
sns.barplot(x='Pclass',y='Survived',hue = 'Sex',data = train) #checking the gender that survived in each Class for better understanding of our data
sns.countplot(x='Embarked',hue = 'Survived',data = train) #checking survived count in embarked table
sns.countplot(x='Embarked',hue = 'Pclass',data = train)
train.head()
train['Name'] #taking out the names from the Name column
train['Title'] = train['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip()) #splitting the title in order to just preserve the title of a person example Miss/Mr etc.
test['Title'] = test['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip()) #splitting the title in order to just preserve the title of a person example Miss/Mr etc.
train['Title'].value_counts() #counting the number of titles in order to adjust those titles that are not present inn  huge numbers
train['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True) #adjusting the tiles properly in order to reduce the number of rows and adjusting the titles properly. This step is for replacing female titles.
test['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
train['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)#adjusting the tiles properly in order to reduce the number of rows and adjusting the titles properly. This step is for replacing male titles.
test['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
sns.barplot(x='Title',y='Survived',data = train) #visualizing the titles  
train['Family'] = train['SibSp'] + train['Parch'] + 1 #making a column family that has the count of family members onboard
test['Family'] = test['SibSp'] + test['Parch'] + 1
train.head()
sns.barplot(x='Family',y='Survived',data = train) #checking the number of people in each family
train['Family'] = pd.cut(train.Family, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big']) #dividing the family in various categories
test['Family'] = pd.cut(test.Family, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
train.head()
y = train['Survived'] #splitting our data
features = ['Pclass', 'Sex', 'Fare', 'Title', 'Embarked', 'Family']
X = train[features]
X.head()
from sklearn.impute import SimpleImputer #importing all of our libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
numerical_cols = ['Fare'] #creating a pipeline in which we can feed our column values and encode these values in order to make it suitable to be used by our classifier. We cannot feed continuous values hence we have to encode them properly.
categorical_cols = ['Pclass', 'Sex', 'Title', 'Embarked', 'Family']


numerical_transformer = SimpleImputer(strategy='median')


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestClassifier(random_state=0, 
                                                               n_estimators=500, max_depth=5))
                             ])

# Preprocessing of training data, fit model 
model.fit(X,y)

print('Cross validation score: {:.3f}'.format(cross_val_score(model, X, y, cv=10).mean()))
X_test = test[features] #our testing data
X_test.head()
preds = model.predict(X_test) #testing our model our test data
result = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds}) #saving our results
result.to_csv('submission4.csv', index=False)
print('Your submission was successfully saved!')
