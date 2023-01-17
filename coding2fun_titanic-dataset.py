# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import tensorflow as tf

import tensorflow.contrib as tflearn



import matplotlib.pyplot as plt



import seaborn as sns

sns.set()



%matplotlib inline



# Any results you write to the current directory are saved as output.
# Load the dataset

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
# Load the dataset

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
# Get some basic information about the dataset

train_data.info()



#It shows, Age, Cabin and Embarked as missing values
# Let's get the count of each features



figure = plt.figure(figsize=(10, 10))

fig_dims = (5, 3)



plt.subplot2grid(fig_dims, (0, 0))

train_data['Survived'].value_counts().plot(kind='bar', title='Survived Count')

plt.subplots_adjust(hspace=.5)



plt.subplot2grid(fig_dims, (0, 1))

train_data['Pclass'].value_counts().plot(kind='bar', title='Passenger Class')

plt.subplots_adjust(hspace=.5)



plt.subplot2grid(fig_dims, (1, 0))

train_data['Sex'].value_counts().plot(kind='bar', title='Gender Count')

plt.subplots_adjust(hspace=.5)



plt.subplot2grid(fig_dims, (1, 1))

train_data['Embarked'].value_counts().plot(kind='bar', title='Embarked Count')

plt.subplots_adjust(hspace=.5)





plt.subplot2grid(fig_dims, (2, 0))

train_data['Age'].value_counts().hist()

plt.title('Age Histogram')

plt.xlabel('Age')

plt.subplots_adjust(hspace=.8)
# Compare each feature with class_Variable

pclass_survived = pd.crosstab(train_data['Pclass'], train_data['Survived'])

pclass_survived
# Normalize pclass_survived

pclass_survived = pclass_survived.div(pclass_survived.sum(axis = 1), axis=0)

pclass_survived
pclass_survived.plot(kind='bar', stacked=True, title='Passenger Class with Survival Rate')

'''

The below plot shows, pepole in 1st class has high survival rate

'''
gender_survived = pd.crosstab(train_data['Sex'], train_data['Survived'])

gender_survived = gender_survived.div(gender_survived.sum(axis = 1), axis = 0)

gender_survived.plot(kind='bar', stacked=True, title='Gender With Survival Rate')

'''

Female is having high survival rate than male

'''
'''

Sex should be converted to numeric

'''

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
'''

We already know, Age is having missing values, Let's fix it here - Fill 

age by median = middle number

'''

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

train_data.info()
'''

Let's Plot the Age

'''

survived_age = pd.crosstab(train_data['Age'], train_data['Survived'])

#sns.boxplot(x = train_data['Survived'], y = train_data['Age'])

sns.boxplot(survived_age)
'''

SibSp and Parch can be merged into one and remove unwanted features

'''

train_data['SibPar'] = train_data['SibSp'] + train_data['Parch']

train_data.drop(['Name', 'Cabin', 'SibSp', 'Parch', 'Ticket', 'Fare', 'PassengerId'], axis=1, inplace=True)

train_data.head(4)
'''

Embarked Value is also having missing values, Fill the embarked val

with mode since it has only 2 missing values

'''

train_data['Embarked'].value_counts()

train_data['Embarked'] = train_data['Embarked'].fillna('S')

train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])

train_data.info()
'''

Preprocess method

'''

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def preprocess(df, keep_passengerid):

    

    # Fill missing values for Age

    df['Age'] = df['Age'].fillna(df['Age'].median())

    

    # Fill missing values for Embarked

    df['Embarked'] = df['Embarked'].fillna('S')

    

    # Convert Nominal to numeric

    df['Pclass'] = label_encoder.fit_transform(df['Pclass'])

    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

    df['Sex'] = label_encoder.fit_transform(df['Sex'])

    

    #Combine SibSp and Parch to one features

    df['SibPar'] = df['SibSp'] + df['Parch']

    

    #Remove 'Name', 'Cabin', 'SibSp', 'Parch', 'Ticket', 'Fare' columns

    df.drop(['Name', 'Cabin', 'SibSp', 'Parch', 'Ticket', 'Fare'], axis = 1, inplace=True)

    

    if not keep_passengerid:

        df.drop(['PassengerId'], axis = 1, inplace=True)        

    return df
features = train_data

target = features['Survived']

features.drop(['Survived'], axis = 1, inplace=True)

train_data.head(3)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size = 0.20, random_state=0)
from sklearn.ensemble import RandomForestClassifier
random_classifier = RandomForestClassifier(n_estimators=100)
model = random_classifier.fit(X=X_train, y=Y_train)

score = model.score(X_train, Y_train)

"Mean accuracy of Random Forest: {0}".format(score)
prediction = model.predict(X_test)
from sklearn.metrics import accuracy_score
'Accuracy : {0}'.format(accuracy_score(Y_test, prediction))
test_data.head(3)
passenger_ids = test_data['PassengerId']

test_data = preprocess(test_data, keep_passengerid=False)

Y_prediction = model.predict(test_data)
submission = pd.DataFrame({

        "PassengerId": passenger_ids,

        "Survived": Y_prediction

    })

submission.to_csv('titanic_Kishore.csv', index=False)