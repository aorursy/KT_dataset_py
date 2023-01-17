# Data Dictionary

# Variable	Definition	Key

# survival	Survival	0 = No, 1 = Yes

# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd

# sex	Sex	

# Age	Age in years	

# sibsp	# of siblings / spouses aboard the Titanic	

# parch	# of parents / children aboard the Titanic	

# ticket	Ticket number	

# fare	Passenger fare	

# cabin	Cabin number	

# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# Variable Notes

# pclass: A proxy for socio-economic status (SES)

# 1st = Upper

# 2nd = Middle

# 3rd = Lower



# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5



# sibsp: The dataset defines family relations in this way...

# Sibling = brother, sister, stepbrother, stepsister

# Spouse = husband, wife (mistresses and fiancés were ignored)



# parch: The dataset defines family relations in this way...

# Parent = mother, father

# Child = daughter, son, stepdaughter, stepson

# Some children travelled only with a nanny, therefore parch=0 for them.
# Importing the usual libraries and filter warnings

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pyplot import xticks

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


#train = pd.read_csv('train.csv')

#test = pd.read_csv('test.csv')



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



print(train.shape,test.shape)

#In the beginning it's important to check the size of your train and test data which later helps in 

#deciding the sample size while testing your model on train data
train.isnull().sum()
test.isnull().sum()
train.info()
train.describe()
train.head()
test.head()
# To see the survival count

ax = sns.countplot("Survived",data=train)

ax.set_title("Survival Count")

for p in ax.patches:

     ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+1))

xticks(rotation=90)

#Looks like the Target variable is not skewed
# To see the survival count among male and female

ax = sns.countplot("Sex",data=train,hue="Survived")

ax.set_title("Looks like more number of female survived than men")

for p in ax.patches:

     ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+1))

xticks(rotation=90)
#Interesting to see how people emabarked for Southamptom suffered the most casualties

ax = sns.countplot("Embarked",data=train,hue="Survived")

for p in ax.patches:

     ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+1))

xticks(rotation=90)

#Looks like the passengers with Pclass=3 surviced the most
ntrain = train.shape[0]

y = train['Survived'].values

df = pd.concat([train.drop(columns=['Survived']), test])

df.shape
df.isnull().sum()
df.groupby(['Pclass','Sex']).mean()
#The above Avg Age by PClass gives a very unique distinction 

#The people who survived in 1st Class were elderlies near 35-43Age group , those in lower classes were young ,

#which could also mean kids survived 
df.loc[(df["Pclass"]==1) & (df["Sex"]=="female") & (df["Age"].isnull()),'Age'] = 37.0

df.loc[(df["Pclass"]==1) & (df["Sex"]=="male") & (df["Age"].isnull()),'Age'] = 41.0



df.loc[(df["Pclass"]==2) & (df["Sex"]=="female") & (df["Age"].isnull()),'Age'] = 27.0

df.loc[(df["Pclass"]==2) & (df["Sex"]=="male") & (df["Age"].isnull()),'Age'] = 30.0



df.loc[(df["Pclass"]==3) & (df["Sex"]=="female") & (df["Age"].isnull()),'Age'] = 22.0

df.loc[(df["Pclass"]==3) & (df["Sex"]=="male") & (df["Age"].isnull()),'Age'] = 25.0
df.isnull().sum()
# Since most of the population onboard was embarked for Southampton and 

#just 2 rows were missing Embarked value so we would fill it with S

df["Embarked"].fillna("S",inplace=True)
# One value of fare is missing

df[df["Fare"].isnull()]
#Lets see whats the average value of fare by a passenger from Pclass = 3 and above the age of 40

df[(df["Pclass"]==3) & (df.Age > 40) & (df.Sex == "male")].mean()
#Fill the missing value with the average

df.loc[(df["Pclass"]==3) & (df.Fare.isnull()) & (df.Sex == "male"), 'Fare'] = 8.5
df.isnull().sum()
# Since almost a thousand people are missing Cabin its best we drop this column as we couldn't see any corelation between the 

# value of Cabin and Survived at this stage

df.drop(columns = ["Cabin"],inplace=True)
#df.dropna(inplace=True)
# Looks like we have dealt with all the missing values quickly 

# Later we could explore multiple other pro techniqies to deal with missing values but at this stage

# we are looking for a quick model solution

df.isnull().sum()
#We feel its better to drop these columns , maybe we could refer to some pro's notebooks to how use them in the model later

df_prep = df.drop(columns=['PassengerId','Name','Ticket'])
df_prep.head()
df_prep.info()
df_prep['Parch'] = df_prep['Parch'].apply(str)

df_prep['SibSp'] = df_prep['SibSp'].apply(str)

df_prep['Pclass'] = df_prep['Pclass'].apply(str)
# Categorical values need to be converted into numbers and one such simple approach is one hot encoding

OneHot = pd.get_dummies(df_prep[['Pclass','Sex','SibSp','Parch','Embarked']])
OneHot.head()
df_prep =  df_prep.drop(columns=['Pclass','Sex','SibSp','Parch','Embarked'])

df_prep = pd.concat([df_prep,OneHot],axis=1)

df_prep.shape
#Once we have cleaned the data we will divide it back again into its orignal format of train and test

df_train = df_prep[:ntrain]

df_test = df_prep[ntrain:]
df_train.head()
# Loading 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
seed = 2

test_size = 0.2



X_train, X_test, y_train, y_test = train_test_split(df_train,y, test_size = test_size , random_state = seed)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# Logistic Regression



model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

f1 = f1_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)

roc_auc = roc_auc_score(y_test,y_pred)

cv_score = cross_val_score(model,X_train,y_train,cv=7,scoring="accuracy")

cv_predict = cross_val_predict(model,X_train,y_train,cv=7)

print('Accuracy: %f' % accuracy)

print('Precision: %f' % precision)

print('Recall: %f' % recall)

print('F1 score: %f' % f1)

print('ROC AUC score: %f' % roc_auc)

print("Accuracy by cross val score : %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))

print("Accuracy by cross val predict : %0.2f" % (accuracy_score(y_train,cv_predict)))
confusion_matrix(y_test,y_pred)
confusion_matrix(y_train,cv_predict)

# Lets try some hyperparameter tuning with Grid search

params = {'C':[0.01,0.1,1,10],

           'penalty':['l1','l2']}

grid_search = GridSearchCV(model,params,cv=10,scoring='accuracy',return_train_score=True)

grid_search.fit(df_train,y)

grid_search.best_params_
model = LogisticRegression(C=10,penalty='l2')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

f1 = f1_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)

roc_auc = roc_auc_score(y_test,y_pred)

cv_score = cross_val_score(model,X_train,y_train,cv=7,scoring="accuracy")

cv_predict = cross_val_predict(model,X_train,y_train,cv=7)

print('Accuracy: %f' % accuracy)

print('Precision: %f' % precision)

print('Recall: %f' % recall)

print('F1 score: %f' % f1)

print('ROC AUC score: %f' % roc_auc)

print("Accuracy by cross val score : %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))

print("Accuracy by cross val predict : %0.2f" % (accuracy_score(y_train,cv_predict)))
#Here is the code to make your submission

test["Survived"] = model.predict(df_test)

submission = test[['PassengerId','Survived']].copy()

submission.to_csv("Titanic_kaggle.csv",index=False)

#Upload the file Titanic_kaggle.csv to the submission