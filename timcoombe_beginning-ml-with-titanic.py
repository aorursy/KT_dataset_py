#Load some useful libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print ("Train Data")

print ("----------")

train.info()

print ("")

print ("Test Data")

print ("----------")

test.info()
testPassengerIds = test[['PassengerId']].copy()

# Replace ages with some random values between the mean and std, rather than just using the mean

# to get a more meaningful distribution



#generate random ages for the train set

train_age_mean   = train["Age"].mean()

train_age_std    = train["Age"].std()

train_age_null_count = train["Age"].isnull().sum()



#generate random ages for the test set

test_age_mean   = test["Age"].mean()

test_age_std    = test["Age"].std()

test_age_null_count = test["Age"].isnull().sum()



train_rand = np.random.randint(train_age_mean - train_age_std, train_age_mean + train_age_std, size = train_age_null_count)

test_rand = np.random.randint(test_age_mean - test_age_std, test_age_mean + test_age_std, size = test_age_null_count)



# fill null values in Age column with random values generated

train["Age"][np.isnan(train["Age"])] = train_rand

test["Age"][np.isnan(test["Age"])] = test_rand



train["Age"] = train["Age"].astype(int)

test["Age"] = test["Age"].astype(int)



#Find the train values for embarked

train['Embarked'].value_counts()
#As there are only 2 missing 'Embarked' values, replace these with the most common value 'S'

train["Embarked"] = train["Embarked"].fillna("S")
#Replace the missing test 'Fare' with the mean

train['Fare'] = train['Fare'].fillna(np.mean(train['Fare']))

test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))
#Use One hot encoding for the 'Embarked' column

train_embarked = pd.get_dummies(train['Embarked'])

train = train.join(train_embarked)



test_embarked = pd.get_dummies(test['Embarked'])

test = test.join(test_embarked)

test.head()
#Function to try and show if there is a correlation between the

#cabin position, denoted by the first character and the survival rate



def calculate_cabin_position(row):

    '''

    function to check if the cabin is null and replace wth the mean from 

    the mean ages dataframe 

    '''

    if pd.isnull(row['Cabin']):

        return 'Z'

    else:

        return row['Cabin'][0]

    

def calculate_cabin_position_known(row):

    

    '''

    function to check if the cabin is known

    '''

    if pd.isnull(row['Cabin']):

        return 0

    else:

        return 1



train['CabinPos'] =train.apply(calculate_cabin_position, axis=1)

train['CabinPosKnown'] =train.apply(calculate_cabin_position_known, axis=1)

test['CabinPos'] =test.apply(calculate_cabin_position, axis=1)

test['CabinPosKnown'] =test.apply(calculate_cabin_position_known, axis=1)



cabinpos_table = pd.crosstab(train['Survived'],train['CabinPos'])

print(cabinpos_table)
## Make the sex an integer

train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)

train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','C','Q','S','CabinPosKnown']]

test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','C','Q','S','CabinPosKnown']]
X = train.drop('Survived', axis = 1)

y = train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth = 3)

classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

print('Training accuracy...', accuracy_score(y_train, classifier.predict(X_train)))

print('Validation accuracy', accuracy_score(y_test, classifier.predict(X_test)))
from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont



with open("tree1.dot", 'w') as f:

     f = tree.export_graphviz(classifier,

                              out_file=f,

                              impurity = False,

                              feature_names = X_test.columns.values,

                              class_names = ['No', 'Yes'],

                              rounded = True,

                              filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])



# Annotating chart with PIL

img = Image.open("tree1.png")

draw = ImageDraw.Draw(img)

img.save('sample-out.png')

PImage("sample-out.png")
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(penalty='l2')



logreg.fit(X_train, y_train)



Y_pred = logreg.predict(X_test)



logreg.score(X_train, y_train)
print('Training accuracy...', accuracy_score(y_train, logreg.predict(X_train)))

print('Validation accuracy', accuracy_score(y_test, logreg.predict(X_test)))
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier()



random_forest.fit(X_train, y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, y_train)
print('Training accuracy...', accuracy_score(y_train, random_forest.predict(X_train)))

print('Validation accuracy', accuracy_score(y_test, random_forest.predict(X_test)))
testPassengerIds['Survived'] = classifier.predict(test).tolist()

testPassengerIds.to_csv('titanic_survivors.csv',index=False)