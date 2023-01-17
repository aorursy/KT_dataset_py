# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
print(train.dtypes)
train.shape
train.columns
print(train.nunique())
train.describe()
train.info()
import seaborn as sns

import matplotlib.pyplot as plt
numeric_variables = ["PassengerId", "Pclass", "Survived", "Age", "SibSp", "Parch", "Fare"]

categorical_variables = ["Name" , "Sex", "Ticket", "Embarked"]
sns.barplot(x = "Pclass" , y = "Survived" , data = train)

#Passenger class 1 people has survived more
Pclass1=train["Survived"][train["Pclass"] == 1].value_counts(normalize=True)[1]*100

Pclass2=train["Survived"][train["Pclass"] == 2].value_counts(normalize=True)[1]*100

Pclass3=train["Survived"][train["Pclass"] == 3].value_counts(normalize=True)[1]*100

Pclass_Values=[Pclass1,Pclass2,Pclass3]

plt.pie(Pclass_Values, labels=('Pclass1','Pclass2','Pclass3') ,explode=(0.1,0.0,0.0), autopct='%1.1f%%')
sns.barplot(x = "Sex" , y = "Survived" , data = train)

#females has survived more
Females=train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True)[1]*100

males=train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True)[1]*100

Sex_values=[males,Females]

plt.pie(Sex_values, labels=('males','Females'),explode=(0.0,0.1),autopct='%1.1f%%')
sns.barplot(x = "SibSp" , y = "Survived" , data = train)

#people with 1 siblings survived more
a = train["Survived"][train['SibSp'] == 0].value_counts(normalize=True)[1]*100

b = train["Survived"][train['SibSp'] == 1].value_counts(normalize=True)[1]*100

c = train["Survived"][train['SibSp'] == 2].value_counts(normalize=True)[1]*100

d = train["Survived"][train['SibSp'] == 3].value_counts(normalize=True)[1]*100

e = train["Survived"][train['SibSp'] == 4].value_counts(normalize=True)[1]*100

Sibling_values=[a,b,c,d,e]

plt.pie(Sibling_values, labels=('SibSp0','SibSp1','SibSp2','SibSp3','SibSp4'),explode=(0.1,0.1,0.1,0.1,0.1), autopct='%1.1f%%')
sns.barplot(x = "Parch" , y = "Survived" , data = train)

#people with parch = 3 survived more
P0=train["Survived"][train["Parch"] == 0].value_counts(normalize=True)[1]*100

P1=train["Survived"][train["Parch"] == 1].value_counts(normalize=True)[1]*100

P2=train["Survived"][train["Parch"] == 2].value_counts(normalize=True)[1]*100

P3=train["Survived"][train["Parch"] == 3].value_counts(normalize=True)[1]*100

P5=train["Survived"][train["Parch"] == 5].value_counts(normalize=True)[1]*100

Parch_values=[P0,P1,P2,P3,P5]

plt.pie(Parch_values, labels=('Parch0','Parch1','Parch2','Parch3','Parch5'),explode=(0.0,0.0,0.0,0.1,0.0),autopct='%1.1f%%')
train.isnull().sum()
missing_value_percentage = print(train.isnull().sum()/len(train))

missing_value_percentage
train = train.drop(["Cabin", "Name", "Fare", "Ticket"], axis = 1)
train.head()
print(train.isnull().sum())



#lets impute other missing values 
#impute age with mean as it is numeric 

#impute embarked with mode as it is categorical

train['Age'] = train['Age'].fillna(train['Age'].mean())

train["Embarked"] = pd.Categorical(train["Embarked"])

train["Embarked"] = train["Embarked"].cat.codes

train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode())

print(train.isnull().sum())

train["Embarked"] = train["Embarked"].astype("object")

print(train.dtypes)
train.head()
train["Sex"] = pd.Categorical(train["Sex"])

train["Sex"] = train["Sex"].cat.codes

train["Sex"] = train["Sex"].astype("object")



#male = 1, female = 0
train.head()
train.shape
train.describe()
numeric_variables2 = ["Pclass", "Survived", "Age",]

categorical_variables2 = [ "Sex", "Embarked"]
import seaborn as sns

import matplotlib.pyplot as plt



for i in numeric_variables2 :

    print(i)

    sns.boxplot(y = train[i])

    plt.xlabel(i)

    plt.ylabel("Values")

    plt.title("Boxplot of " + i)

    plt.show()
# Identify outliers

#calculate Inner Fence, Outer Fence, and IQR



for i in numeric_variables2:

    print(i)

    q75, q25 = np.percentile(train.loc[:,i], [75, 25])

    iqr = q75 - q25

    Innerfence = q25 - (iqr*1.5)

    Upperfence = q75 + (iqr*1.5)

    print("Innerfence= "+str(Innerfence))

    print("Upperfence= "+str(Upperfence)) 

    print("IQR ="+str(iqr))

    



# replace outliers with NA



    train.loc[train[i]<Innerfence, i] = np.nan

    train.loc[train[i]>Upperfence, i] = np.nan
print(train.isnull().sum()/len(train))
#impute age with mean

train['Age'] = train['Age'].fillna(train['Age'].mean())

print(train.isnull().sum())        
train.head()
train.dtypes
numeric_variables3 = ["PassengerId", "Survived", "Pclass", "Age", "SibsSp", "Parch"]
Correlation = train.loc[:, numeric_variables3]

correlation_result = Correlation.corr()

print(correlation_result)

    
heatmap =  sns.heatmap(correlation_result)



#No collinearity is found
#Data Distribution



x = train.drop(["Survived"], axis = 1)

y = train["Survived"]
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split 



#divide the data into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)
from sklearn.ensemble import RandomForestClassifier



RM_model = RandomForestClassifier(n_jobs = 100, n_estimators = 100, random_state = 123)

RM_model.fit(x_train, y_train)
RM_model.score(x_test, y_test)
y_predrm = RM_model.predict(x_test)



#Accuracy



accuracy_rm = round(accuracy_score(y_predrm, y_test)*100,2)

print(accuracy_rm)
from sklearn.metrics import classification_report  

print(classification_report(y_predrm, y_test))
from sklearn.metrics import confusion_matrix 

print(confusion_matrix(y_predrm, y_test))
from sklearn.linear_model import LogisticRegression

LR_Model = LogisticRegression()

LR_Model.fit(x_train, y_train)
LR_Model.score(x_test,y_test)
y_predlr = LR_Model.predict(x_test)



#Accuracy

accuracy_lr = round(accuracy_score(y_predlr, y_test) * 100, 2)

print("Accuracy:",accuracy_lr)
print(classification_report(y_predlr, y_test))
print(confusion_matrix(y_predlr, y_test))
test.head()
#Drop unnecessary columns 

test = test.drop(['Name','Ticket','Fare','Cabin'], axis=1)

test.head()
#Convert datatype

test["Sex"] = pd.Categorical(test["Sex"])

test["Sex"] = test["Sex"].cat.codes

test["Sex"] = test["Sex"].astype("object")



test["Embarked"] = pd.Categorical(test["Embarked"])

test["Embarked"] = test["Embarked"].cat.codes

test["Embarked"] = test["Embarked"].astype("object")



test.head()



#Check for NA values

test.isnull().sum()
test["Age"] = test["Age"].fillna(test["Age"].mean())
test.isnull().sum()
test.head()
test["Survived"] = RM_model.predict(test)
test.head()
predicted_test = test.drop(['Pclass','Sex','Age','SibSp', "Parch", "Embarked"], axis=1)

y_output = predicted_test["Survived"]

given_output = pd.read_csv("../input/gender_submission.csv")
y_given= given_output["Survived"]
accuracy_final = round(accuracy_score(y_output, y_given) * 100, 2)

print("Accuracy:",accuracy_final)
print(confusion_matrix(y_output, y_given))
sns.countplot(x="Survived",data=test , )
test['Survived'].value_counts()
predicted_test.to_csv('submission.csv', index=False)