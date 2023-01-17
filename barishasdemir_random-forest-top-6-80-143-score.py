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
# Import matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/titanic/train.csv")



display(train.head())



print(train.info())

print(train.info())

print(train.describe())
# Visualize with a countplot

sns.countplot(x="Survived", data=train)

plt.show()



# Print the proportions

print(train["Survived"].value_counts(normalize=True))
# Visualize with a countplot

sns.countplot(x="Pclass", hue="Survived", data=train)

plt.show()



# Proportion of people survived for each class

print(train["Survived"].groupby(train["Pclass"]).mean())



# How many people we have in each class?

print(train["Pclass"].value_counts())
# Display first five rows of the Name column

display(train[["Name"]].head())


# Get titles

train["Title"] = train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



# Print title counts

print(train["Title"].value_counts())
# Print the Surviving rates by title

print(train["Survived"].groupby(train["Title"]).mean().sort_values(ascending=False))
# Print the missing values in Age column

print(train["Age"].isnull().sum())
# Survived by age

sns.distplot(train[train.Survived==1]["Age"],color="y", bins=7, label="1")



# Death by age

sns.distplot(train[train.Survived==0]["Age"], bins=7, label="0")

plt.legend()

plt.title("Age Distribution")

plt.show()
# Visualize with a countplot

sns.countplot(x="Sex", hue="Survived", data=train)

plt.show()



# Proportion of people survived for each class

print(train["Survived"].groupby(train["Sex"]).mean())



# How many people we have in each class?

print(train["Sex"].value_counts())
print(train["SibSp"].value_counts())



print(train["Parch"].value_counts())



train["family_size"] = train["SibSp"] + train["Parch"]



print(train["family_size"].value_counts())



# Proportion of people survived for each class

print(train["Survived"].groupby(train["family_size"]).mean().sort_values(ascending=False))

# Print the first five rows of the Ticket column

print(train["Ticket"].head(15))
# Get first letters of the tickets

train["Ticket_first"] = train["Ticket"].apply(lambda x: str(x)[0])



# Print value counts

print(train["Ticket_first"].value_counts())



# Surviving rates of first letters

print(train.groupby("Ticket_first")["Survived"].mean().sort_values(ascending=False))
# Print 3 bins of Fare column

print(pd.cut(train['Fare'], 3).value_counts())



# Plot the histogram

sns.distplot(train["Fare"])

plt.show()



# Print binned Fares by surviving rate

print(train['Survived'].groupby(pd.cut(train['Fare'], 3)).mean())
# Print the unique values in the Cabin column

print(train["Cabin"].unique())



# Get the first letters of Cabins

train["Cabin_first"] = train["Cabin"].apply(lambda x: str(x)[0])



# Print value counts of first letters

print(train["Cabin_first"].value_counts())



# Surviving rate of Cabin first letters

print(train.groupby("Cabin_first")["Survived"].mean().sort_values(ascending=False))
# Make a countplot

sns.countplot(x="Embarked", hue="Survived", data=train)

plt.show()



# Print the value counts

print(train["Embarked"].value_counts())



# Surviving rates of Embarked

print(train["Survived"].groupby(train["Embarked"]).mean())
# Load the train and the test datasets

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



print(test.info())
# Put the mean into the missing value

test['Fare'].fillna(train['Fare'].mean(), inplace = True)
from sklearn.impute import SimpleImputer

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



# Imputers

imp_embarked = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

imp_age = IterativeImputer(max_iter=100, random_state=34, n_nearest_features=2)



# Impute Embarked

train["Embarked"] = imp_embarked.fit_transform(train[["Embarked"]])

test["Embarked"] = imp_embarked.transform(test[["Embarked"]])



# Impute Age

train["Age"] = np.round(imp_age.fit_transform(train[["Age"]]))

test["Age"] = np.round(imp_age.transform(test[["Age"]]))
from sklearn.preprocessing import LabelEncoder



# Initialize a Label Encoder

le = LabelEncoder()



# Encode Sex

train["Sex"] = le.fit_transform(train[["Sex"]].values.ravel())

test["Sex"] = le.fit_transform(test[["Sex"]].values.ravel())
# Family Size

train["Fsize"] = train["SibSp"] + train["Parch"]

test["Fsize"] = test["SibSp"] + test["Parch"]
# Ticket first letters

train["Ticket"] = train["Ticket"].apply(lambda x: str(x)[0])

test["Ticket"] = test["Ticket"].apply(lambda x: str(x)[0])



# Cabin first letters

train["Cabin"] = train["Cabin"].apply(lambda x: str(x)[0])

test["Cabin"] = test["Cabin"].apply(lambda x: str(x)[0])
# Titles

train["Title"] = train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

test["Title"] = test['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
# Group the family_size column

def assign_passenger_label(family_size):

    if family_size == 0:

        return "Alone"

    elif family_size <=3:

        return "Small_family"

    else:

        return "Big_family"

    

# Group the Ticket column

def assign_label_ticket(first):

    if first in ["F", "1", "P", "9"]:

        return "Ticket_high"

    elif first in ["S", "C", "2"]:

        return "Ticket_middle"

    else:

        return "Ticket_low"

    

# Group the Title column    

def assign_label_title(title):

    if title in ["the Countess", "Mlle", "Lady", "Ms", "Sir", "Mme", "Mrs", "Miss", "Master"]:

        return "Title_high"

    elif title in ["Major", "Col", "Dr"]:

        return "Title_middle"

    else:

        return "Title_low"

    

# Group the Cabin column  

def assign_label_cabin(cabin):

    if cabin in ["D", "E", "B", "F", "C"]:

        return "Cabin_high"

    elif cabin in ["G", "A"]:

        return "Cabin_middle"

    else:

        return "Cabin_low"
# Family size

train["Fsize"] = train["Fsize"].apply(assign_passenger_label)

test["Fsize"] = test["Fsize"].apply(assign_passenger_label)



# Ticket

train["Ticket"] = train["Ticket"].apply(assign_label_ticket)

test["Ticket"] = test["Ticket"].apply(assign_label_ticket)



# Title

train["Title"] = train["Title"].apply(assign_label_title)

test["Title"] = test["Title"].apply(assign_label_title)



# Cabin

train["Cabin"] = train["Cabin"].apply(assign_label_cabin)

test["Cabin"] = test["Cabin"].apply(assign_label_cabin)
train = pd.get_dummies(columns=["Pclass", "Embarked", "Ticket", "Cabin","Title", "Fsize"], data=train, drop_first=True)

test = pd.get_dummies(columns=["Pclass", "Embarked", "Ticket", "Cabin", "Title", "Fsize"], data=test, drop_first=True)
target = train["Survived"]

train.drop(["Survived", "SibSp", "Parch", "Name", "PassengerId"], axis=1, inplace=True)

test.drop(["SibSp", "Parch", "Name","PassengerId"], axis=1, inplace=True)
display(train.head())

display(test.head())



print(train.info())

print(test.info())
from sklearn.model_selection import train_test_split



# Select the features and the target

X = train.values

y = target.values



# Split the data info training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34, stratify=y)
# Import Necessary libraries

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report





"""

# Initialize a RandomForestClassifier

rf = RandomForestClassifier(random_state=34)



params = {'n_estimators': [50, 100, 200, 300, 350],

          'max_depth': [3,4,5,7, 10,15,20],

          'criterion':['entropy', 'gini'],

          'min_samples_leaf' : [1, 2, 3, 4, 5, 10],

          'max_features':['auto'],

          'min_samples_split': [3, 5, 10, 15, 20],

          'max_leaf_nodes':[2,3,4,5],

          }



clf = GridSearchCV(estimator=rf,param_grid=params,cv=10, n_jobs=-1)



clf.fit(X_train, y_train.ravel())



print(clf.best_estimator_)

print(clf.best_score_)



"""



# Initialize a RandomForestClassifier

rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=5, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=350,

                       n_jobs=None, oob_score=False, random_state=34, verbose=0,

                       warm_start=False)



rf.fit(X_train, y_train)



# Predict from the test set

y_pred = rf.predict(X_test)



# Predict from the train set

y_pred_train = rf.predict(X_train)



# Print the accuracy with accuracy_score function

print("Accuracy Train: ", accuracy_score(y_train, y_pred_train))



# Print the accuracy with accuracy_score function

print("Accuracy Test: ", accuracy_score(y_test, y_pred))



# Print the confusion matrix

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))
# Create a pandas series with feature importances

importance = pd.Series(rf.feature_importances_,index=train.columns).sort_values(ascending=False)



sns.barplot(x=importance, y=importance.index)

# Add labels to your graph

plt.xlabel('Importance')

plt.ylabel('Feature')

plt.title("Important Features")

plt.show()
last_clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=5, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=350,

                       n_jobs=None, oob_score=True, random_state=34, verbose=0,

                       warm_start=False)



last_clf.fit(train, target)

print("%.4f" % last_clf.oob_score_)
# Store passenger ids

ids = pd.read_csv("/kaggle/input/titanic/test.csv")[["PassengerId"]].values



# Make predictions

predictions = last_clf.predict(test.values)



# Print the predictions

print(predictions)



# Create a dictionary with passenger ids and predictions

df = {'PassengerId': ids.ravel(), 'Survived':predictions}



# Create a DataFrame named submission

submission = pd.DataFrame(df)



# Display the first five rows of submission

display(submission.head())



# Save the file

submission.to_csv("submission.csv", index=False)