# import necessary libraries first



# pandas to open data files & processing it.

import pandas as pd



# numpy for numeric data processing

import numpy as np



# sklearn to do preprocessing & ML models

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Matplotlob & seaborn to plot graphs & visulisation

import matplotlib.pyplot as plt 

import seaborn as sns



# to fix random seeds

import random, os



# ignore warnings

import warnings

warnings.simplefilter(action='ignore')
titanic_data = pd.read_csv("../input/titanic/train.csv")

titanic_data.shape
titanic_data.head()
titanic_data.describe()
# Survival

titanic_data['Survived'].value_counts()
# Ticket class

titanic_data['Pclass'].value_counts()
# Gender

titanic_data['Sex'].value_counts()
# Siblings

titanic_data['SibSp'].value_counts()
# Parent or Childs

titanic_data['Parch'].value_counts()
# Embarked station

titanic_data['Embarked'].value_counts()
sns.countplot(titanic_data['Sex']);
sns.barplot(titanic_data['Survived'], titanic_data['Sex']);
sns.barplot(titanic_data['Survived'], titanic_data['Fare'], titanic_data['Pclass']);
sns.boxplot(x=titanic_data["Fare"])

plt.show()
# Only take rows which have "Fare" value less than 250.

titanic_data = titanic_data[titanic_data['Fare'] < 250]

titanic_data.shape
sns.boxplot(x=titanic_data["Age"])

plt.show()
titanic_data.isna().sum()
titanic_data.drop("Cabin", axis=1, inplace=True)

titanic_data.shape
titanic_data.columns
age_mean = titanic_data['Age'].mean()

print(age_mean)
titanic_data['Age'].fillna(age_mean, inplace=True)
titanic_data.isna().sum()
titanic_data['Embarked'].value_counts()
titanic_data['Embarked'].fillna("S", inplace=True)
titanic_data.isna().sum()
titanic_data.head(10)
titanic_data['total_family_members'] = titanic_data['Parch'] + titanic_data['SibSp'] + 1



# if total family size is 1, person is alone.

titanic_data['is_alone'] = titanic_data['total_family_members'].apply(lambda x: 0 if x > 1 else 1)



titanic_data.head(10)
sns.barplot(titanic_data['total_family_members'], titanic_data['Survived'])
sns.barplot(titanic_data['is_alone'], titanic_data['Survived'])
def age_to_group(age):

    if 0 < age < 12:

        # children

        return 0

    elif 12 <= age < 50:

        # adult

        return 1

    elif age >= 50:

        # elderly people

        return 2

    

titanic_data['age_group'] = titanic_data['Age'].apply(age_to_group)

titanic_data.head(10)
sns.barplot(titanic_data['age_group'], titanic_data['Survived']);
titanic_data['name_title'] = titanic_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)

titanic_data.head()
titanic_data['name_title'].value_counts()
def clean_name_title(val):

    if val in ['Rev', 'Col', 'Mlle', 'Mme', 'Ms', 'Sir', 'Lady', 'Don', 'Jonkheer', 'Countess', 'Capt']:

        return 'RARE'

    else:

        return val



titanic_data['name_title'] = titanic_data['name_title'].apply(clean_name_title)

titanic_data['name_title'].value_counts()
sns.barplot(titanic_data['name_title'], titanic_data['Survived']);
titanic_data.head(10)
# save the target column 

target = titanic_data['Survived'].tolist()



titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1, inplace=True)
titanic_data.head()
le = preprocessing.LabelEncoder()

titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'])

titanic_data['Embarked'] = le.fit_transform(titanic_data['Embarked'])

titanic_data['name_title'] = le.fit_transform(titanic_data['name_title'])

titanic_data.head()
train_data, val_data, train_target, val_target = train_test_split(titanic_data, target, test_size=0.2)

train_data.shape, val_data.shape, len(train_target), len(val_target)
def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)



# We fix all the random seed so that, we can reproduce the results.

seed_everything(2020)
# Train the LogisticRegression model.



model = LogisticRegression()

model.fit(train_data, train_target)
# Predict labels on Validation data which model have never seen before.



val_predictions = model.predict(val_data)

len(val_predictions)
# first 10 values of validation_predictions

val_predictions[:10]
# Calculate the accuracy score on validation data.

# We already have correct target information for them.



accuracy = accuracy_score(val_target, val_predictions)

accuracy
print("We got %.3f percent accuracy on our validation unseen data !!"%(accuracy*100))

print("We are #.3f correct in predicting whether a person will survice in Titanic crash !!"%(accuracy*100))