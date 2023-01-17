import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
os.listdir("../input/titanic")
train_data = pd.read_csv("../input/titanic/train.csv")
train_data.head()
train_data.info()
sns.set_style(style='whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train_data);
sns.countplot(x='Survived',hue='Pclass',data=train_data);
sns.distplot(train_data['Age'].dropna(),kde=False,bins=30);
sns.countplot(train_data['SibSp'])
train_data.isnull().sum()
#Analysing the age and passenger class through boxplot
plt.figure(figsize=(10,8))
sns.boxplot(x='Pclass',y='Age',data=train_data)
def add_age(col):
    Age = col[0]
    Pclass = col[1]
    if pd.isnull(Age):
        #fill the null columns with the average value from the boxplot
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
train_data['Age'] = train_data[['Age','Pclass']].apply(add_age,axis=1)
train_data.isnull().sum()
train_data.drop('Cabin',axis=1,inplace=True)
train_data.dropna(inplace=True)
train_data.isnull().sum()
train_data.head()
sex = pd.get_dummies(train_data['Sex'],drop_first=True)
sex.head()
embark = pd.get_dummies(train_data['Embarked'],drop_first=True)
embark.head()
train_data = pd.concat([train_data,sex,embark],axis=1)
train_data.head()
train_data.columns
train_data = train_data.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1)
train_data.head()
test = pd.read_csv("../input/titanic/test.csv")
test_data = test.copy()
test_data.head()
test_data.isnull().sum()
plt.figure(figsize=(10,8))
sns.boxplot(x='Pclass',y='Age',data=test_data)
def add_age_test(col):
    Age = col[0]
    Pclass = col[1]
    if pd.isnull(Age):
        #fill the null columns with the average value from the boxplot
        if Pclass == 1:
            return 42
        elif Pclass == 2:
            return 26
        else:
            return 24
    else:
        return Age
test_data['Age'] = test_data[['Age','Pclass']].apply(add_age_test,axis=1)
test_data.drop('Cabin',axis=1,inplace=True)
test_data['Fare'].mean()
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
test_data.isnull().sum()
test_data.head(155)
sex_test = pd.get_dummies(test_data['Sex'],drop_first=True)
sex_test.head()
embark_test = pd.get_dummies(test_data['Embarked'],drop_first=True)
embark_test.head()
test_data = pd.concat([test_data,sex_test,embark_test],axis=1)
test_data.head()
test_data = test_data.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1)
test_data.head()
X_train = train_data.drop(['Survived'],axis=1)
y_train = train_data['Survived'].values
X_test = test_data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
predictions.shape
real_values = pd.read_csv("../input/titanic/gender_submission.csv")
real_values.shape
from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(real_values['Survived'].values,predictions)
print(classification_report(real_values['Survived'].values,predictions))
result = pd.DataFrame(predictions,columns=['Survived'])
result = pd.concat([test['PassengerId'],result],axis=1)
result
result.to_csv("submission.csv",index=False)
os.listdir("../working")