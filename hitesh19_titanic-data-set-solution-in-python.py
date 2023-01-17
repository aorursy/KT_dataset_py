# import the packges library
import numpy as np
import pandas as pd
import seaborn as sns
sns.set() 
#import the data set.
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')

# display the top row of data
titanic_train.head()
# shape of dataset
titanic_train.shape
#display the columns of train dataset.
titanic_train.columns
# display the describe the data exclude the categorical values
titanic_train.describe()
# display the descirbe include the categorical data
titanic_train.describe(include = 'all')
titanic_train.info()
#Name, Sex,Cabin,Embarked ,Ticket,have Categorical variable and other variable have numerice variable.
# find out the missing value in train dataset.
titanic_train.isnull().sum()
#Age have 177 , cabin 687, and Embarked 2 missing values.

# now let do #now we Exploratory Data Analysis (EDA) with Visualization test data of titanic.

titanic_test.head()
 #Survived column is not present in Test data. We have to train our classifier using the Train data and generate predictions (Survived) on Test data
titanic_test.shape
# 418 have observation and 11 columns . one columns Survived is not present in test data.


titanic_test.info()
#Name, Sex,Cabin,Embarked ,Ticket,have Categorical variable and other variable have numerice variable.

# findout the missing values in test data
titanic_test.isnull().sum()
# Age 86 , Cabin 327,Fare 1 missing values

# now we are visualize the realtionship between depended variable, indepeded variable on Train data.
# findout the how many persentage people survived or not survived.

Count_Survived = titanic_train[titanic_train['Survived'] == 1]
Count_not_Survived = titanic_train[titanic_train['Survived'] == 0]
print ("Survived: %i (%.1f%%)"%(len(Count_Survived), float(len(Count_Survived))/len(titanic_train)*100.0))
print(" Not Survived: %i (%.1f%%)"%(len(Count_not_Survived),float(len(Count_not_Survived))/len(titanic_train)*100))

# findout the pclass vs Suvival
titanic_train.Pclass.value_counts()

titanic_train[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()

sns.barplot(x='Pclass', y='Survived', data=titanic_train)
# reation with Sex variable to Suvived variable
titanic_train['Sex'].value_counts()
# now we do find the how many females and males are survived or not.
titanic_train.groupby('Sex').Survived.value_counts()

titanic_train[['Sex','Survived']].groupby('Sex',as_index =False).mean()
sns.barplot(x='Sex', y='Survived', data=titanic_train)
pd.crosstab(titanic_train['Sex'],titanic_train['Survived'])
# imputing the missing values
titanic_train.isnull().sum()
# Age have 177 missing value is numerical variable replace by mean value
titanic_train['Age'].fillna(int(titanic_train['Age'].mean()),inplace = True)

titanic_train.isnull().sum()
titanic_train['Cabin'].fillna(titanic_train['Cabin'].mode()[0],inplace = True)
titanic_train.isnull().sum()
titanic_train['Embarked'].fillna(titanic_train['Embarked'].mode()[0],inplace = True)
titanic_train.isnull().sum()
# now lets imputing the missing value in test data
titanic_test.isnull().sum()
#Age have 86 missing values
titanic_test['Age'].fillna(int(titanic_test['Age'].mean()),inplace = True)
titanic_test.isnull().sum()
titanic_test['Cabin'].fillna(titanic_test['Cabin'].mode()[0],inplace = True)
titanic_test['Fare'].fillna(titanic_train['Fare'].mean(),inplace = True)
titanic_test.isnull().sum()

titanic_train.info()
titanic_train.head(100)
#We drop unnecessary columns/features and keep only the useful ones for our experiment.
drop_categorical = ['Name','PassengerId','Ticket','Cabin']
titanic_train = titanic_train.drop(drop_categorical,axis = 1)


titanic_train.head()
drop_categorical = ['Name','PassengerId','Ticket','Cabin']
titanic_test= titanic_test.drop(drop_categorical,axis = 1)
titanic_test.head()
titanic_train.head()
from sklearn import preprocessing 
colname  = ['Sex','Embarked']
le = {}
for x in colname:
    le[x] =   preprocessing.LabelEncoder()
    
for x in colname:
    titanic_train[x] = le[x].fit_transform(titanic_train[x])
    
    
titanic_train.head()
#we will now create set of dependant and independant variables
#creating/splitting training and testing datasets and running model
#here in X_train we are taking data from loan id to property_area i.e all independant vars
X_train = titanic_train.iloc[:,1:].values
Y_train = titanic_train.iloc[:,0].values
titanic_test.head()
from sklearn import preprocessing 
colname  = ['Sex','Embarked']
le = {}
for x in colname:
    le[x] =   preprocessing.LabelEncoder()
    
for x in colname:
    titanic_test[x] = le[x].fit_transform(titanic_test[x])
    
titanic_test.head()
X_test = titanic_test.iloc[:,:].values
# now create model by logstic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
model= classifier.fit(X_train,Y_train)
y_pred = model.predict(X_test)

# now we check the accurecy of model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
acc_lg = round(model.score(X_train, Y_train) * 100, 2)

print(str(acc_lg)  +  '  percent')

# now we check in SVM.
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
svc_model.fit(X_train,Y_train)
y_pred = svc_model.predict(X_test)
acc_svm = round(svc_model.score(X_train, Y_train) * 100, 2)
print(str(acc_svm)  +  '  percent')
# now we check in Random forest  model 
from sklearn.ensemble import RandomForestClassifier
model_RandomForest=RandomForestClassifier(501)
model12= model_RandomForest.fit(X_train,Y_train)
Y_pred=model_RandomForest.predict(X_test)
acc_rfc = round(model12.score(X_train,Y_train) * 100,2)
print(str(acc) + '  percent')

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
model2 = dtc.fit(X_train,Y_train)
y_pred = m.predict(X_test)
acc_dt = round(model2.score(X_train,Y_train)* 100 ,2)
print(str(acc) + ' precent')
#Comparing Models
#Let's compare the accuracy score of all the classifier models used above.
model_list = pd.DataFrame({
            'Model' : ['Logistic Regression','Support Vector Machines','Decision Tree','Random Forest'],
             'Score' :[acc_lg,acc_svm,acc_rfc,acc_dt]
})
model_list
# finally we got the accuracy score in above table Decision Tree,Random Forest have highest score.