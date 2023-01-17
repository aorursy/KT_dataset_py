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



#matplot and seaborn for visualization 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#read the train csv file

train = pd.read_csv("/kaggle/input/titanic/train.csv")

#print first 5 rows of dataset

train.head()
#read the test csv file

test = pd.read_csv("/kaggle/input/titanic/test.csv")

#print first 5 rows of dataset

test.head()
#dimesnions of train data

train.shape
#Describes the summary of train dataset in statistical info

train.describe()
#dimensions of test data

test.shape
##Describes the summary of test dataset in statistical info

test.describe()
#Missing value
#Count number of Nan values in train dataset

train_nullvalue_count=train.isnull().sum()

train_nullvalue_count
#Counts number of Nan values in test dataset

test_nullvalue_count=test.isnull().sum()

test_nullvalue_count
#percentage of Nan values in train data

null_train=train.isnull().sum()/len(train)*100

null_train
#Percentage of Nan values in test data

null_test=test.isnull().sum()/len(test)*100

null_test
#Complete view of Nan values in count and percentage

df = pd.DataFrame({'Train_null_count':train_nullvalue_count,'Train_Null_percent':null_train,'Test_Null_count':test_nullvalue_count,'Test_Null_percent_test':null_test})

df
#Passengers Survived and not survived in a ship 

#1 - Not survived 2- Survived

sns.factorplot('Survived',data=train,kind='count',palette='Blues')
#We can see the passengers survived in Different Classes

sns.countplot('Survived',hue='Pclass',data=train,palette='coolwarm')
sns.countplot(x='Survived',hue='Sex',data=train,palette='coolwarm')
sns.countplot(x='Survived',hue='Embarked',data=train,palette='coolwarm')
sns.barplot(x='Sex',y='Survived',hue='Pclass',data=train,palette='GnBu_d')
sns.boxplot(x='Survived',y='Age',data=train,palette='Blues')

sns.stripplot(x='Survived',y='Age',jitter=True,data=train)
train['Embarked'].value_counts().plot(kind='bar')

plt.title('Boarding Places')

plt.show()
sns.factorplot(x='Pclass',y='Survived',data=train,color='r')
sns.factorplot(x='Embarked',y='Survived',data=train,color='r')
#Correlation amoung features

corr=train.corr()

f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr,square=True,annot=True, linewidths=.5, ax=ax)
#Values missing in train dataset

#age values were missing in train data

#we compute with mean and fill those Nan values with the mean

mean_age = train['Age'].mean() 

train['Age'] = train['Age'].fillna(value=mean_age)
#Embarked feature is a char column, so we normalize the feature and check which category has large scale of percent values 

train['Embarked'].value_counts(normalize=True)
#Fill the Nan value with 'S' category because it is highest repeated category value in Embarked feature

train = train.fillna({'Embarked':'S'})
#Replace the Nan values of age with average age value in test data 

mean_test_age = test['Age'].mean()

test['Age'] = test['Age'].fillna(value=mean_test_age)
#Replace the Nan values of Fare with average Fare value in test data

mean_fare = test['Fare'].mean()

test['Fare'] = test['Fare'].fillna(value=mean_fare)
#Embarked feature is a char column, so we normalize the feature and check which category has large scale of percent values

test['Embarked'].value_counts(normalize=True)
#Fill the Nan value with 'S' category because it is highest repeated category value in Embarked feature

test = test.fillna({'Embarked':'S'})
#visualizate the train dataset by heatmap, to see the where the data was missing

sns.heatmap(train.isnull())
#visualize the test dataset by heatmap, to see the where we the data was missing 

sns.heatmap(test.isnull())
#drop the unwanted columns 

train = train.drop(['Name','Cabin','Ticket'],axis=1)
test = test.drop(['Cabin','Name','Ticket'],axis=1)
train.head()
test.head()
#create the category feature with one hot encoding 

sex = pd.get_dummies(train['Sex'],drop_first=True)
embarked = pd.get_dummies(train['Embarked'],drop_first=True)
#After encoding the features drop the repective column 

train = train.drop(['Sex','Embarked'],axis=1)
#concat the encoding columns into train dataset

train = pd.concat([train,sex,embarked],axis=1)
train.head()
#create the category feature with one hot encoding

sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embarked_test = pd.get_dummies(test['Embarked'],drop_first=True)
#After encoding the features drop the repective column

test = test.drop(['Sex','Embarked'],axis=1)
#concat the encoding columns into train dataset

test = pd.concat([test,sex_test,embarked_test],axis=1)
test.head()
#Feature Scaling

#We can see that Age, Fare are measured on different scales, so we need to do Feature Scaling first before we proceed with predictions. 

#If we dont scale down the feature then these columns will have high importance while modeling, to overcome this we use standardscaler, 

#so all columns will have equal importance
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()

train[['Age','Fare']] = std_scale.fit_transform(train[['Age','Fare']])
#Split the features into predictors and target variable
x = train.drop(['Survived','PassengerId'],axis=1)

y = train['Survived']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=45)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
lr = LogisticRegression()

lr.fit(x_train,y_train)
pred_lr = lr.predict(x_test)
lr_score=cross_val_score(lr,x_train,y_train,scoring='accuracy',cv=10)

lr_score
lr_score.mean()
lr_accuracy=accuracy_score(y_test,pred_lr)

lr_accuracy
print(classification_report(y_test,pred_lr))
#76% of area is under the curve and it is quite good in this model
roc_auc_score(y_test,pred_lr)
gbc = GradientBoostingClassifier(n_estimators=250,verbose=-1)

gbc.fit(x_train,y_train)
pred_gbc = gbc.predict(x_test)
gbc_score=cross_val_score(gbc,x_train,y_train,scoring='accuracy',cv=10)

gbc_score
gbc_score.mean()
gbc_accuracy=accuracy_score(y_test,pred_gbc)

gbc_accuracy
print(classification_report(y_test,pred_gbc))
#80% of area is under the curve and it is quite good in this model
roc_auc_score(y_test,pred_gbc)
abc = AdaBoostClassifier()

abc.fit(x_train,y_train)
pred_abc = abc.predict(x_test)
abc_score=cross_val_score(abc,x_train,y_train,scoring='accuracy',cv=10)

abc_score
abc_score.mean()
abc_accuracy=accuracy_score(y_test,pred_abc)

abc_accuracy
#78% of area is under the curve and it is quite good in this model
roc_auc_score(y_test,pred_abc)
print(classification_report(y_test,pred_abc))
rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)
pred_rfc = rfc.predict(x_test)
rfc_score=cross_val_score(rfc,x_train,y_train,scoring='accuracy',cv=10)

rfc_score
#80% of area is under the curve and it is quite good in this model
rfc_score.mean()
rfc_accuracy=accuracy_score(y_test,pred_rfc)

rfc_accuracy
roc_auc_score(y_test,pred_rfc)
print(classification_report(y_test,pred_rfc))
#Accuracy for the models
score = pd.DataFrame({'Models':['LogesticRegression_score','GradientBoostingClassifier_score','AdaBoostClassifier_score','RandomForestClassifier'],'Scores':[lr_accuracy,gbc_accuracy,abc_accuracy,rfc_accuracy]})

score.sort_values('Scores', ascending=False)
test_pred = pd.DataFrame(pred_gbc, columns= ['Survived'])
new_test = pd.concat([test, test_pred], axis=1, join='inner')
submission = pd.DataFrame({

        'PassengerId' : test['PassengerId'],

        'Survived' : test_pred['Survived']})



submission.to_csv('titanic_submission.csv', index=False)