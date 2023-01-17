#import os
#print(os.listdir("../input"))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/train.csv")
train.head(4)
train.info()
train.isnull().sample(25)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex', data=train, palette='RdBu_r')
sns.countplot(x='Survived',hue='Pclass', data=train)
sns.distplot(train['Age'].dropna(), kde=False, bins=30 )
sns.countplot(x='SibSp', data=train)
train['Fare'].hist(bins=50, figsize=(10,4))
plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass', y='Age', data=train)
def impute_average(columns):
    Age = columns[0]
    Pclass = columns[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_average, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train.drop('Cabin', axis=1, inplace=True)
train.head(3)
#there are few missing values to get rid of them we drop them.
train.dropna(inplace=True)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train,sex,embark],axis=1)
train.head(3)
train.drop(['Sex', 'Embarked', 'Name','Ticket'],axis=1, inplace=True)
train.head(3)
train.drop('PassengerId', axis=1, inplace=True)
train.head(3)
X = train.drop('Survived', axis=1)
y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
sns.heatmap(confusion_matrix(y_test, predictions),annot=True,fmt="d",cmap='ocean_r' ,robust=True)
# Lastly we can strengthen the results with cross_val_score
from sklearn.model_selection import cross_val_score

print(cross_val_score(model, X, y, cv=19).mean())