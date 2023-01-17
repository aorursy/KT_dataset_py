from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
train = pd.read_csv('../input/titanic/train.csv')
train.name = 'Train Data'
test = pd.read_csv('../input/titanic/test.csv')
test.name = 'Test Data'
# to play with data I create copies of them

train_copy = train.copy(deep=True)
train_copy.name = 'Train Data'
test_copy = train.copy(deep=True)
test_copy.name = 'Test Data'
train.head(6)
# Printing datasets' info

for data in [train, test]:
    print('Info of %s \n' %data.name), 
    print(data.info())
    print('\n')
    
# In test data one column is missing - 'Survived'
# It will be the feature which we will want to predict
# Describing train data

train_copy.describe()
data_cleaner = [train, test]
# Seeing null values in datasets

for data in data_cleaner:
    print('Null values in %s'%data.name), 
    print('in every column: \n')
    print(data.isnull().sum())
    print('\n')
# Heatmap to see null values - training data

plt.figure(figsize=(10, 6))

sns.heatmap(data=train.isnull(), cmap='plasma', yticklabels=False, cbar=False)
plt.show()
# Viewing Age column 

plt.figure(figsize=(10, 6))
plt.title('Age distribution in every class', fontsize=15)
sns.boxenplot(x='Pclass', y='Age', data=train, palette='GnBu_d')
# Removing Cabin, Ticket and PassengerId column

train.drop(columns=['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test.drop(columns=['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)
# For Embarked for now I decide to replace NaN values with 'S'

train['Embarked'].fillna(value='S', inplace=True)
# For Age column I decide to see age distribution and deicde which mean value assign
# I have to see how age is connected with Pclass

plt.figure(figsize=(10, 6))
plt.title('Age distribution', fontsize=15)
sns.distplot(train['Age'].dropna(), kde=True, bins=40)
# For mean age for every class i want to replace null values with mean age for specific class
# I am preparing a funcition 

class_mean_age = pd.DataFrame(train.groupby('Pclass')['Age'].mean())
class_mean_age
def mean_age(col):
    age = col[0]
    pclass = col[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 30
        else:
            return 25
    else:
        return age

    
# Applying function to Age column to set mean values for missing ones    
train['Age'] = train[['Age', 'Pclass']].apply(mean_age, axis=1)
test['Age'] = test[['Age', 'Pclass']].apply(mean_age, axis=1)
# For test dataset one values is missing also in Fare column. I decided to replace this value with mean value of Fare which is 32

test.fillna(value=32, inplace=True)
train.head(6)
# Visualisation 1 - survived people in each class
# Result: overwhelmingly more people from third class died in disaster. 

plt.figure(figsize=(10, 6))
plt.title('Number of survived people versus classes', fontsize=15)
sns.countplot(data=train, x='Pclass', hue='Survived', palette='Blues')

# Number of dead people in every class - i want to sum and print percentage of people
class3 = train[(train_copy['Pclass'] == 3) & (train_copy['Survived'] == 0)].count()['Pclass']
class2 = train[(train_copy['Pclass'] == 2) & (train_copy['Survived'] == 0)].count()['Pclass']
class1 = train[(train_copy['Pclass'] == 1) & (train_copy['Survived'] == 0)].count()['Pclass']

sum_dead = class3+class2+class1
class1_dead = round((class1/sum_dead)*100, 2)

print('Percentage of people from First Class who died is: %s' %class1_dead),
print('%')
# Visualisation 2 - survived people in exact age
# Result: as we can see there is blue peak for survived babies and kids.
# Also there is more older people who survivde (30-60 years) but in age 70-80 year more people died.

plt.figure(figsize=(10, 6))
plt.title('Survived people vs Age', fontsize=15)
g = sns.kdeplot(train['Age'][train['Survived']==0], color='red', shade=True)
g = sns.kdeplot(train['Age'][train['Survived']==1], color='blue', shade=True)
g.set_xlabel('Age')
g.set_ylabel('Frequency')
g.legend(['Not Survived', 'Survived'])
# Visualisation 3 - survived people based on gender
# Result: in every class more females survived than males. 

sns.catplot(data=train, x='Pclass', y='Survived', hue='Sex', palette='GnBu_d', kind='bar')
plt.title('Distribution of Survival based on Gender', fontsize=15)
plt.ylabel('Survival Probability')
# Mean values of females and males who survived
# Result: much more females survived during the disaster

train[['Sex', 'Survived']].groupby('Sex').mean()
# Visualisation 4 - Embarked and survived categorical plot
# Result: people who embarked in Cherbourg had more chance to survive

plt.figure(figsize=(10, 6))
plt.title('Survived people with specific place of embarkation', fontsize=15)
plt.xlabel('Survival probability')
sns.barplot(data=train, x='Embarked', y='Survived', palette='GnBu_d')
# Table shows descending survival rate versus place of embark

train[['Embarked', 'Survived']].groupby('Embarked').mean().sort_values(by='Survived', ascending=False)
# Visualisation 5 - siblings/spouses abord
# Result: one or two sibling/spuses had more chance to survive

plt.figure(figsize=(10, 6))
plt.title('Survived vs sibling/spouses aboard', fontsize=15)
sns.barplot(data=train, x='SibSp', y='Survived', palette='GnBu_d')
train_corr = train.corr()

plt.figure(figsize=(10, 6))
plt.title('Correlations between features', fontsize=15)
sns.heatmap(train_corr, cmap='Blues', annot=True, linewidths=.5)

# As we can see the biggest correlation is between Survived&Fare, Fare&SibSp and Parch&Fare
train.head()
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train.head()
sex = pd.get_dummies(train['Sex'], drop_first=True)
embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embarked], axis=1)
sex = pd.get_dummies(test['Sex'], drop_first=True)
embarked = pd.get_dummies(test['Embarked'], drop_first=True)
test = pd.concat([test, sex, embarked], axis=1)
train.drop(['Sex', 'Embarked'], axis=1, inplace=True)
test.drop(['Sex', 'Embarked'], axis=1, inplace=True)
train.rename(columns={'male': 'Sex'}, inplace=True)
test.rename(columns={'male': 'Sex'}, inplace=True)
train.head(6)
test.head()
# Preparing X features and y value to predict for training and test data

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
y_predictions = logmodel.predict(X_test)
# Accuracy = (TP+TN)/total

acc_matrix = round((133+74)/268, 2)
acc_matrix
# Error rate = (FP+FN)/total

error_matrix = round((40+21)/268,2)
error_matrix
# Printing accuracy

accuracy_log = round(logmodel.score(X_train, y_train) * 100, 2)
accuracy_log
# Printing correlations

coef_df = pd.DataFrame(train.columns[1:])
coef_df.columns = ['Feature']
coef_df
coef_df['Correlation'] = pd.Series(logmodel.coef_[0])
coef_df.sort_values(by='Correlation', ascending=False)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_predictions = knn.predict(X_test)

accuracy_knn = round(knn.score(X_train, y_train) * 100, 2)
accuracy_knn