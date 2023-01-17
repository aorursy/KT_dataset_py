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
# Importing necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

from sklearn import metrics

from sklearn.metrics import mean_squared_error, make_scorer,confusion_matrix, r2_score,classification_report

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB,BernoulliNB

from sklearn.svm import SVC
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

full_data=(train,test)
train.head()
# common statistical approach

train.describe()
# we have pandas profiling lib which is basically give us over all analysing report

import pandas_profiling as pd_prof

data_report=pd_prof.ProfileReport(train)

data_report
# to find out number of null values in each of the data sets

print(train.isna().sum())

print(test.isna().sum())
plt.figure(1)

plt.subplot(131)

train.Pclass.plot(kind='hist',bins=50,figsize=(18,6),title='Pclass')

plt.subplot(132)

train.Fare.plot(kind='hist',bins=50,title='Fare',color='r')
# To know the number of non-null entries in each column and total number of entries

train.info()
train.skew()
# Checking co relation

corr=train.corr()

corr
# checking death vs age relation

survive_pclass=train.groupby(['Survived','Pclass'])['Age'].agg(['mean','count','median'])

survive_pclass
print(train.groupby('Survived')['Age'].mean())

#print(data.groupby('Survived')['Age'].mode())

print(train.groupby('Survived')['Age'].median())
# grouping the data by survivors and siblings/spouses, and finding the meant, count and median of age

survive_SibSp=train.groupby(['Survived','SibSp'])['Age'].agg(['mean','count','median'])

survive_SibSp
survive_SibSp=train.groupby('Survived')['Embarked'].value_counts(normalize=True)

survive_SibSp
# Checking co relation

corr=train.corr()

sns.heatmap(corr)
# creating a user defined function which will filter the rows according to whether or not those people have survived



def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    # creating data frame from the bifurcation(alive/dead) we have done above

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    #plottinf a stacked bar chart of the dataframe created

    df.plot(kind='bar',stacked=True, figsize=(10,5))
#lets check class-wise feature of the data set



bar_chart('Pclass')
# As there are various designation in the data set, so we are extracting the name using regular expressions

for i in full_data:

    i['Title']=i.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



# replacing the variations of a similar designation with one label, like Miss for Mlle, Mme, Ms

for i in full_data:

    i['Title']=i['Title'].replace('Mlle','Miss')

    i['Title']=i['Title'].replace('Mme','Miss')

    i['Title']=i['Title'].replace('Ms','Miss')

    i['Title']=i['Title'].replace(['Don','Rev','Dona','Lady','Sir','Col','Countess', 'Jonkheer'],'Rare')

    i['Title']=i['Title'].replace(['Dr','Capt','Major',],'Master')

# Checking the dead/alive status according to designation

bar_chart('Title')
# Label encoding the designation

maped_title= {'Miss':1,'Mrs':2,'Mr':3,'Master':4,'Rare':5}

for i in full_data:

    i['Title']=i['Title'].map(maped_title)
bar_chart('Sex')
# Label Encoding the gender column 

sex_maped = {"male": 0, "female": 1}

for i in full_data:

    i['Sex'] = i['Sex'].map(sex_maped)
# filling the missing values in age column with median

for i in full_data:

    i['Age']=i['Age'].fillna(i['Age'].median())

    

#Creating bins age wise

for i in full_data:

    i['New_Age']=i['Age'].apply(lambda x: 'Teen' if x<=18 else('Adult' if 19 <= x <=35 else ('Aged' if 36<=x<55 else 'Old')))

    

    

bar_chart('New_Age')
# Label Encoding the Age column, the bins that were created 

age_maped={'Teen':1,'Adult':2,'Aged':3,'old':4}

for i in full_data:

    i['New_Age']=i['New_Age'].map(age_maped)
# finding out the family members of a person

# for this we will find the sum of number of siblings/spouses and parents/children + the person themselves

for i in full_data:

    i['Family']=i['SibSp']+i['Parch']+1

    

'''family_maped = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for i in full_data:

    i['FamilySize'] = i['FamilySize'].map(family_maped)'''

# creating bins into Small, Medium and Big as per the number of family members

for i in full_data:

    i['Family']=i['Family'].apply(lambda x: 'Small' if x<=1 else('Medium' if 2 <= x <=4 else 'Big' ))

    



bar_chart('Family')
# Label encoding the bins in family feature

family_map={'Small':1,'Medium':2,'Big':3}

for i in full_data:

    i['Family']=i['Family'].map(family_map)
# grouping the data according to dead or alive and observing the count of family members 

relatives_survival=train.groupby('Survived')['Family'].value_counts()

relatives_survival
# filling missing values in Fare column

for i in full_data:

    i['Fare']=i['Fare'].fillna(i['Fare'].median())

# Creting bins for Fare column for visualization

for i in full_data:

    i['Fare_new']=i['Fare'].apply(lambda x: 'lower' if x<=50.0 else ('middle' if 51.0 <= x <= 100.0 else 'high' ))



bar_chart('Fare_new')
# Label encoding fare

fare_maped={'lower':1,'middle':2,'high':3}

for i in full_data:

    i['Fare_new']=i['Fare_new'].map(fare_maped)
for i in full_data:

    i['Cabin'] = i['Cabin'].str[:1]
# diving the cabin column according to Class and creating the dataframe class-wise

Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
cabin_maped = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

#cabin_maped = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}

for i in full_data:

    i['Cabin'] = i['Cabin'].map(cabin_maped)
#Misisng value

for i in full_data:

    i['Cabin']=i['Cabin'].fillna(i['Cabin'].median())
bar_chart('Embarked')
# filling missing values in the Embarked column and label encoding it 

for i in full_data:

    i['Embarked'] = i['Embarked'].fillna('S')

embarked_maped = {"S": 0, "C": 1, "Q": 2}

for i in full_data:

    i['Embarked'] =i['Embarked'].map(embarked_maped)
# To know what all columns are left with missing values

train.isna().sum(),test.isna().sum()
#test['Title']=test['Title'].fillna(test['Title'].median())

test['New_Age']=test['New_Age'].fillna(test['New_Age'].median())

train['New_Age']=train['New_Age'].fillna(train['New_Age'].median())

test.isna().sum(),train.isna().sum()
# droping columns

col_drop = ['Ticket', 'SibSp', 'Parch','Name','Age','Fare']

train = train.drop(col_drop, axis=1)

test = test.drop(col_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)
# Splitting the given training data set into test and train

x=train.iloc[:,1:]

y=train['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
KNeighbors_Model=KNeighborsClassifier()

KNeighbors_Model.fit(x_train,y_train)

y_predict_KNeighbors=KNeighbors_Model.predict(x_test)

rmse_KNeighbors=(mean_squared_error(y_test,y_predict_KNeighbors))

r2_KNeighbors=r2_score(y_test,y_predict_KNeighbors)

print("Accuracy is", metrics.accuracy_score(y_test, y_predict_KNeighbors))

print('RMSE IS',rmse_KNeighbors)

print('R-SQUARE',r2_KNeighbors)

cm = confusion_matrix(y_test, y_predict_KNeighbors)

print("Confusion Matrix is :", cm, sep='\n')

report_KNeighbors = classification_report(y_test,y_predict_KNeighbors)

print(report_KNeighbors)
DecisionTree_Model=DecisionTreeClassifier()

DecisionTree_Model.fit(x_train,y_train)

y_predict_DecisionTree=DecisionTree_Model.predict(x_test)

rmse_DecisionTree=(mean_squared_error(y_test,y_predict_DecisionTree))

r2_DecisionTree=r2_score(y_test,y_predict_DecisionTree)

print("Accuracy is", metrics.accuracy_score(y_test, y_predict_DecisionTree))

print('RMSE IS',rmse_DecisionTree)

print('R-SQUARE',r2_DecisionTree)

cm = confusion_matrix(y_test, y_predict_DecisionTree)

print("Confusion Matrix is :", cm, sep='\n')

report_DecisionTree = classification_report(y_test,y_predict_DecisionTree)

print(report_DecisionTree)
SVM_Model=SVC()

SVM_Model.fit(x_train,y_train)

y_predict_SVM=SVM_Model.predict(x_test)

rmse_SVM=(mean_squared_error(y_test,y_predict_SVM))

r2_SVM=r2_score(y_test,y_predict_SVM)

print("Accuracy is", metrics.accuracy_score(y_test, y_predict_SVM))

print('RMSE IS',rmse_SVM)

print('R-SQUARE',r2_SVM)

cm = confusion_matrix(y_test, y_predict_SVM)

print("Confusion Matrix is :", cm, sep='\n')

report_SVM = classification_report(y_test,y_predict_SVM)

print(report_SVM)
test_data = test.drop("PassengerId", axis=1).copy()

prediction = SVM_Model.predict(test_data)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()