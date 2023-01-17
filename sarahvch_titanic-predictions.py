import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

df = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df.head()
df.describe()
sns.heatmap(df.isnull(),yticklabels=False)
sns.boxplot(x='Pclass', y='Age', data=df)
print('First Class', df[df['Pclass'] == 1]['Age'].mean())

print('Second Class', df[df['Pclass'] == 2]['Age'].mean())

print('Third Class', df[df['Pclass'] == 3]['Age'].mean())
def impute_age(cols):

    """

    This function looks for blank age values and then fills in that missing value with 

    the average age for the passengers class.

    """

    Age = cols[0]

    Class = cols[1]

    if pd.isnull(Age):

        if Class == 1:

            return 38

        elif Class == 2:

            return 30

        else:

            return 25

    else:

        return Age
df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df.head()
df_feat = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

df_feat.drop(['Survived'], axis=1, inplace=True)
df_feat.head()
X = df_feat

y = df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

lg.fit(X_train, y_train)
pred = lg.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(classification_report(y_test, rf_pred))

print(confusion_matrix(y_test, rf_pred))