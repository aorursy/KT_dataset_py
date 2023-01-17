# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

"""
Import required packages
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()

from sklearn.preprocessing import Imputer

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
train_df= pd.read_csv("../input/train.csv")
test_df= pd.read_csv("../input/test.csv")
'''
#EDA

checking train and test date frame values
1. how many columns existed in test and train datasets
2. Any difference in data types
3. How many missing values present in train and test data frames
4. check distributions of integer and categorical values
5. Strategy for filling missing values
6. Use simple graphs to understand more about data

'''

'''
#Feature Engineering

1. extract information from Name column and created Title column
2. Age -> child
3. SibSp + Parch  -> is_alone
4. SibSp + Parch  -> family_group
5. Age -> age_group
6. 

'''
train_df.head()
train_df.columns.values
test_df.head()
train_df.describe()
print(train_df.dtypes)
test_df.dtypes
train_df.columns.values
train_df.Sex.value_counts()
test_df.columns.values

print(train_df.describe())
print("*"*20)
print(test_df.describe())
print(train_df.info())
print("*"*20)
print(test_df.info())
print(train_df.shape)
print('*'*20)
print(test_df.shape)
'''
train_df contains 891 observations and 12 variables

test_df contains 418 observations and 11 variables

only difference is 1 column (Survived)
'''
print(train_df.isnull().sum())
print('*'*20)
print(test_df.isnull().sum())
print(train_df.isnull().sum())
print('*'*20)
test_df.isnull().sum()
import missingno as mn

mn.matrix(train_df)
mn.matrix(test_df)
'''
train_df: 3 columns has missing values
Age, Cabin and Embarked

test_df: 3 columns has missing values
Age, Fare and Cabin
'''
'''
test_df Fare column has only 1 missing value, so decided to fill with mean value
'''
'''
saving PassengerId column from test_df 

dropping PassengerId from train and test data frames
'''

full_df = train_df.append(test_df) 
passenger_id=test_df['PassengerId']


train_df.drop(['PassengerId'], axis=1, inplace=True)
test_df.drop(['PassengerId'], axis=1, inplace=True)
test_df.shape
sns.boxplot(x='Survived',y='Fare',data=train_df)
train_df[train_df.Fare > 400]
test_df[test_df.Fare > 400]
'''
Removed outliers from Fare column 
'''
train_df=train_df[train_df['Fare']<400]
'''
Sex column is category column, so assigning female = 0 and male = 1 for both data frames
'''
train_df['Sex'] = train_df.Sex.apply(lambda x: 0 if x == "female" else 1)
test_df['Sex'] = test_df.Sex.apply(lambda x: 0 if x == "female" else 1)
train_df.head()
'''
Filling missing values in Fare column with mean value
'''
pd.options.display.max_columns = 99
test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)
train_df.head()
'''
Extract information from Name column
'''
for name_string in full_df['Name']:
    full_df['Title']=full_df['Name'].str.extract(' ([A-Za-z]+)\.',expand=True)
print(full_df.Title.value_counts())
#print('\n')
print('\n' + '*'*20 + '\n')
print(full_df.Title.value_counts(normalize=True))
print('\n' + '*'*20 + '\n')
full_df.Title.value_counts(normalize=True).plot('bar')
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
full_df.replace({'Title': mapping}, inplace=True)
print(full_df.Title.value_counts())
print('\n' + '*'*20 + '\n')
print(full_df.Title.value_counts(normalize=True))
print('\n' + '*'*20 + '\n')
full_df.Title.value_counts(normalize=True).plot('bar')
train_df['Title']=full_df['Title'][:891]
test_df['Title']=full_df['Title'][891:]


full_df.head()
print(full_df.Title.value_counts(normalize=True).plot('bar'))
print(sns.barplot(x='Title',y='Survived',data=train_df))
'''
Planning to impute missing Age values based on Title group
'''

age_to_impute = full_df.groupby('Title')['Age'].median()
age_to_impute
'''
Filling missing values using Title groupby and median
'''
titles=['Mr','Miss','Mrs','Master','Rev','Dr']
for title in titles:
    age_to_impute = full_df.groupby('Title')['Age'].median()[titles.index(title)]
    #print(age_to_impute)
    full_df.loc[(full_df['Age'].isnull()) & (full_df['Title'] == title), 'Age'] = age_to_impute
full_df.isnull().sum()
train_df['Age']=full_df['Age'][:891]
test_df['Age']=full_df['Age'][891:]
print(test_df.isnull().sum())
print('\n' + '*'*20 + '\n')
train_df.isnull().sum()
train_df.describe()
train_df.groupby('Survived').mean()
train_df.groupby('Sex').mean()
train_df.corr()
plt.subplots(figsize = (15,8))
sns.heatmap(train_df.corr(), annot=True,cmap="Blues")
plt.title("Correlations Among Features", fontsize = 20)

#sns.heatmap(df, cmap="YlGnBu")
#sns.heatmap(df, cmap="Blues")
#sns.heatmap(df, cmap="BuPu")
#sns.heatmap(df, cmap="Greens")
sns.barplot(x = "Sex", y = "Survived", data=train_df)
plt.title("Survived/Non-Survived Passenger Gender Distribution")
labels = ['Female', 'Male']
plt.ylabel("% of passenger survived")

plt.xticks(sorted(train_df.Sex.unique()), labels)
ax=sns.countplot(x='Sex',data=train_df,hue='Survived',linewidth=2)
train_df.shape

plt.title('Survived vs Not-survived')
plt.xlabel('Gender')
plt.ylabel("No. of Passenger Survived")
labels = ['Female', 'Male']

plt.xticks(sorted(train_df.Survived.unique()),labels)

'''
get legend and assign back values

'''
leg = ax.get_legend()
leg.set_title('Survived')
legs=leg.texts
legs[0].set_text('No')
legs[1].set_text('Yes')
ax=sns.countplot(x='Pclass',hue='Survived',data=train_df)
plt.title("Passenger Class Distribution - Survived vs Non-Survived")
leg=ax.get_legend()
leg.set_title('Survival')
legs=leg.texts

legs[0].set_text('No')
legs[1].set_text("yes")
plt.subplots(figsize=(10,8))
sns.kdeplot(train_df.loc[(train_df['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')
ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived' )

labels = ['First', 'Second', 'Third']
plt.xticks(sorted(train_df.Pclass.unique()),labels)
plt.subplots(figsize=(10,8))
ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 0),'Fare'],color='r',shade=True,label='Not Survived')
ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 1),'Fare'],color='b',shade=True,label='Survived' )
plt.title('Fare Distribution Survived vs Non Survived')
plt.ylabel('Frequency of Passenger Survived')
plt.xlabel('Fare')
train_df.head()
fig,axs=plt.subplots(figsize=(10,8))

sns.kdeplot(train_df.loc[(train_df['Survived']==0),'Age'],color='r',shade=True,label='Not Survived')
sns.kdeplot(train_df.loc[(train_df['Survived']==1),'Age'],color='b',shade=True,label='Survived')
train_df.head()
'''
SibSp + Parch + 1 -> family_size
'''

train_df['family_size'] = train_df.SibSp + train_df.Parch+1
test_df['family_size'] = test_df.SibSp + test_df.Parch+1
print(train_df.family_size.value_counts())
print('\n' + '*'*20 + '\n')
print(train_df.family_size.value_counts(normalize=True))
print('\n' + '*'*20 + '\n')
print(train_df.family_size.value_counts(normalize=True).plot(kind='bar'))
'''
Creating family_group (alone, small, large) based on family_size column
'''


def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a

train_df['family_group'] = train_df['family_size'].map(family_group)
test_df['family_group'] = test_df['family_size'].map(family_group)
train_df.head()
'''
Creating one more column (is_alone) based on family_size column
'''

train_df['is_alone'] = [1 if i<2 else 0 for i in train_df.family_size]
test_df['is_alone'] = [1 if i<2 else 0 for i in test_df.family_size]
train_df.head()
'''
Creating child column based on Age. If Age < 16 defining as child
'''


train_df['child'] = [1 if i<16 else 0 for i in train_df.Age]
test_df['child'] = [1 if i<16 else 0 for i in test_df.Age]
train_df.child.value_counts()
train_df.head()
'''
creating calculated_fare per person basedon Fare and family_size
'''

train_df['calculated_fare'] = train_df.Fare/train_df.family_size
test_df['calculated_fare'] = test_df.Fare/test_df.family_size
train_df.head()
train_df.calculated_fare.mean()
train_df.calculated_fare.mode()
print(train_df.calculated_fare.min())
train_df.calculated_fare.max()
plt.subplots(figsize=(10,8))

ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 0),'calculated_fare'],color='r',shade=True,label='Not Survived')
ax=sns.kdeplot(train_df.loc[(train_df['Survived'] == 1),'calculated_fare'],color='b',shade=True,label='Survived' )
plt.title('Fare Distribution Survived vs Non Survived')
plt.ylabel('Frequency of Passenger Survived')
plt.xlabel('Fare')
def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a
train_df['fare_group'] = train_df['calculated_fare'].map(fare_group)
test_df['fare_group'] = test_df['calculated_fare'].map(fare_group)
train_df.head()
'''
Creating dummy variables for the columns ('Title',"Pclass",'Embarked', 'family_group', 'fare_group') and droping frist 
to avoid dummy variable trap

Droping columns ('Cabin', 'family_size','Ticket','Name', 'Fare')
We already created new different columns based on family_size, Name and Fare

I feel Cabin and Ticket columns not giving much information so droping these also
'''

train_df = pd.get_dummies(train_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)
train_df.drop(['Cabin', 'family_size','Ticket','Name', 'Fare'], axis=1, inplace=True)
test_df.drop(['Ticket','Name','family_size',"Fare",'Cabin'], axis=1, inplace=True)
train_df.head()
def age_group_fun(age):
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 4: 
        a = 'toddler'
    elif age <= 13:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 35:
        a = 'Young_Adult'
    elif age <= 45:
        a = 'adult'
    elif age <= 55:
        a = 'middle_aged'
    elif age <= 65:
        a = 'senior_citizen'
    else:
        a = 'old'
    return a
'''
Creating age_group based on Age column
'''

train_df['age_group'] = train_df['Age'].map(age_group_fun)
test_df['age_group'] = test_df['Age'].map(age_group_fun)
'''
Creating dummy varibles for age_group column
'''

train_df = pd.get_dummies(train_df,columns=['age_group'], drop_first=True)
test_df = pd.get_dummies(test_df,columns=['age_group'], drop_first=True)


'''
Dropoing Age and calculated_fare columns
'''

train_df.drop(['Age','calculated_fare'],axis=1,inplace=True)
test_df.drop(['Age','calculated_fare'],axis=1,inplace=True)
train_df.head()
test_df.head()
columns = train_df.columns.values
for column in columns:
    print(column)
    print(train_df[column].value_counts())
    print(train_df[column].value_counts(normalize=True))
X = train_df.drop('Survived', 1)
y = train_df['Survived']
''' 
Importing necessary modules for creating our models
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
'''
Applying standardscalar for all variables to standize

'''

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
testframe = std_scaler.fit_transform(test_df)
testframe.shape
'''
Spliting data into train and test 
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=45)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score,recall_score,confusion_matrix
'''
Checking LogisticRegressing with basic tuning parameters
'''

logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(X_train,y_train)
predict=logreg.predict(X_test)
'''
Checking accuracy
'''

print(accuracy_score(y_test,predict))
print(confusion_matrix(y_test,predict))
print(precision_score(y_test,predict))
print(recall_score(y_test,predict))

'''
Using GridSerchCV finding best tuning parameters
'''

C_vals = [0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0]
penalties = ['l1','l2']

param = {'penalty': penalties, 'C': C_vals }
grid = GridSearchCV(logreg, param,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=10,shuffle=True), n_jobs=1,scoring='accuracy')
grid.fit(X_train,y_train)
print (grid.best_params_)
print (grid.best_score_)
print(grid.best_estimator_)
'''
Applying best estimators to LogisticRegression
'''

logreg_grid = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])
logreg_grid.fit(X_train,y_train)
y_pred = logreg_grid.predict(X_test)
logreg_accy = round(accuracy_score(y_test, y_pred), 3)


'''
Checking accuracy again
'''

print (logreg_accy)
print(confusion_matrix(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
'''
Trying RamomForest
'''

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=20,max_features=0.2, min_samples_leaf=8,random_state=20)



randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
random_accy = round(accuracy_score(y_pred, y_test), 3)
'''
Checking accuracy metrics
'''

print (random_accy)
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
test_df2= pd.read_csv("../input/test.csv")
y_test_pred = randomforest.predict(testframe)
y_test_pred
submission = pd.DataFrame(pd.DataFrame({
        "PassengerId": test_df2.PassengerId,
        "Survived": y_test_pred
    }))


submission.to_csv("submission_4.csv", index = False)


