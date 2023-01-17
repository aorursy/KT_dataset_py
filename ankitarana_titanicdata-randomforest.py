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
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head(3)
train_data.info()
# age and embarked, CABIN HAS MISSING VALUES
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
sns.heatmap(train_data.isnull())
train_data.isnull().sum()
# most values in cabin are null,so it can be dropped 
# the values in age can be imputed with mean
# impute ['Name', 'Sex', 'Ticket', 'Embarked'] with mode(categorical ) also

train_df=train_data.drop('Cabin',axis=True)
# before imputation lets encode our data to numerica data from categorical data,by identifying the categorical data
categorical_col=train_df.select_dtypes(include=['object']).columns
categorical_col
ar_of_modes=np.array(train_df[categorical_col].mode())
val=ar_of_modes[0]
val
f=0
for i in categorical_col:
    train_df[i]=train_df[i].fillna(val[f])
    f+=1
train_df.isnull().sum()
# now impute age with median
train_df['Age']=train_df['Age'].fillna(train_df['Age'].median())
print(train_df.isnull().sum())
sns.heatmap(train_df.isna())
# now we have treated all the missing data,let's explore the data

sns.pairplot(train_df.drop(["PassengerId", "Parch", "SibSp"], axis = 1), hue = "Survived")
sns.countplot(train_df['Sex'])
sns.countplot(x='Survived', hue='Sex', data=train_df)
sns.countplot(x='Survived', hue='Embarked', data=train_df)
# finding correlatons in data
train_df.corr()

#creating heatmap
plt.figure(figsize=(20,18))
cor=train_df.corr()
sns.heatmap(cor,annot=True)
plt.show()
#correlation with output variable
cor_target=abs(cor['Survived'])
print(cor_target.sort_values())
for i in categorical_col:
    train_df[i]=pd.factorize(train_df[i])[0]

feature_col=train_df.drop(['Survived','PassengerId'],axis=1).columns

X=train_df[feature_col]
y=train_df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test_data))
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
#writing the accuracy score
lr_score = classifier.score(X_test, y_test)
print(lr_score)
predictions = classifier.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
test_PassengerId = test_data['PassengerId']
test_df= test_data.drop(["PassengerId",'Cabin'], axis=1)
categorical_col=test_df.select_dtypes(include=['object']).columns
categorical_col
test_df.columns
ar_of_modes=np.array(test_df[categorical_col].mode())
val=ar_of_modes[0]
val

f=0
for i in categorical_col:
    test_df[i]=test_df[i].fillna(val[f])
    f+=1
    
test_df['Age']=test_df['Age'].fillna(test_df['Age'].median())
   
for i in categorical_col:
    test_df[i]=pd.factorize(test_df[i])[0]
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].median())
   
test_survived = pd.Series(classifier.predict(test_df), name = "Survived").astype(int)
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results.to_csv("submission.csv", index = False) #output

results['Survived'].value_counts()
results.head(100)
results.describe()
