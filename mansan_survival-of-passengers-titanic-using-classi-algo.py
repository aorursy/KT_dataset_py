# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from sklearn import preprocessing

import sklearn.linear_model as skl_lm

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.pylab as pylab

pylab.rcParams[ 'figure.figsize' ] = 8 , 6
man_train=pd.read_csv('../input/train.csv')

#Reading a train data file

man_train.head()
#Verifying null values from the dataset.

man_train.isnull().any()
#Verifying indexs and rows

print( man_train.shape)
# Age , Cabin , Embarked features are having null values, So verifying how many null values in each features

man_train.isnull().sum()
# Taking backup of the Original dataset and start cleaning the data.

man_imp=man_train

print (round(man_imp['Cabin'].isnull().sum()/man_imp.shape[0]*100,2),"% of Null Value in CABIN feature ")

print (round(man_imp['Age'].isnull().sum()/man_imp.shape[0]*100,2),"% of Null Value in Age feature ")

print (round(man_imp['Embarked'].isnull().sum()/man_imp.shape[0]*100,2),"% of Null Value in Embarked  feature ")
# So We are going to fill null as mean value for age null values

man_imp['Age']=man_imp.Age.fillna(man_imp.Age.mean())

man_imp.isnull().sum()
# Cabin feature having more than 70% of Null value , So we are dropping the feature as this not going to help anyway.

man_imp=man_imp.drop(['Cabin'],axis=1)
# Let's see what are all the classes( Ticket Class) present in the Embarked feature.

man_imp.groupby(['Embarked']).size()
# Let's see Age wise passanger counts sample.

man_imp.groupby(['Age']).size().sort_values(ascending=False).head()
# No of Men/Women survived from the train data set.

man_imp['Survived'].groupby(man_imp['Sex']).size()
fel=man_imp['Survived'].groupby(man_imp['Sex']).size()[0]/man_imp['Sex'].shape[0]*100

mal=man_imp['Survived'].groupby(man_imp['Sex']).size()[1]/man_imp['Sex'].shape[0]*100

total=man_imp['Sex'].shape[0]







print ("="*50)

print (" Total number of passenger in the ship was :",total)

print (" Male Passengers  ",round(mal,2),"%  ")

print (" FeMale Passengers  ",round(fel,2),"%  ")

print ("="*50)
plt.ylabel('Count')

plt.title('Gender details')

man_imp.groupby(man_imp['Sex']).size().plot.bar()
# Passenger Class wise survived details

sns.countplot(y='Pclass',hue='Survived',data=man_imp)
# Gender wise survived details

sns.countplot(y='Sex',hue='Survived',data=man_imp)
# Age wise passangers details

plt.hist(x='Age',data=man_imp)

plt.xlabel('AGE')

plt.ylabel('Count')
# selecting features which is datatype object based

man_imp.select_dtypes(include =['object']).columns
# selecting features which is datatype float based



man_imp.select_dtypes(include =['float']).columns
man_imp.columns
man_imp.head()
#dropping PassengerId & Name as these are not needed for model selection.

man_imp=man_imp.drop(['PassengerId'],axis=1)

man_imp=man_imp.drop(['Name'],axis=1)
man_imp.head()
man_imp=man_imp.drop(['Ticket'],axis=1)
man_imp.head()
man_imp.shape
man_imp.isnull().sum()
# here in Embarked feature  null value is only 2 so we are update these 2 null value to "S" ( which Embarked class having major value).

man_imp['Embarked']=man_imp.Embarked.fillna('S')
man_imp.isnull().sum()
man_imp.groupby(['Embarked']).size()
man_imp.head()
# Here feature Sex & Embarked class are not numarical values so we have to covert this to numberical value using dummies.

man_dum=pd.get_dummies(man_imp)
man_dum.head()
man_dum.corr()
# Heatmap 

sns.heatmap(man_dum.corr(),annot=True,cmap="Blues")
# As everything is over now we are going to start model selection.

# Spliting train & test data.

X=man_dum.loc[:,man_dum.columns != 'Survived']

Y=man_dum.loc[:,man_dum.columns == 'Survived']
print ("X dataset shape",X.shape)

print ("Y dataset shape",Y.shape)
#Doing Train , test splitup

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=6)
print ("X_train dataset shape",X_train.shape)

print ("Y_train dataset shape",Y_train.shape)

print ("="*30)

print ("X_test dataset shape",X_test.shape)

print ("Y_test dataset shape",Y_test.shape)
# As this is a classification dataset ( have to predict Survuived / Not  ) We will try all possible algorithms





# 1 - Logistic Algorithm



model_logi=skl_lm.LogisticRegression()

model_logi.fit(X_train,Y_train.values.ravel())

# Fitted in Algorithm with X_train & Y_train, now going to predict the X_test values. 

#Then compare the predictied value with Y_test value so that we can ensure how best we predicted.

Pre_Logi=model_logi.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



# For ensure the prediction percentage we are using classification_report , confusion_matrix & accuracy_score here.

print(classification_report(Y_test,Pre_Logi))
confusion_matrix(Y_test, Pre_Logi)
log_ac=round(accuracy_score(Y_test, Pre_Logi),2)*100

print("Logistic Alg Accuracy :",log_ac,"%")
# 2 - DecisionTree







from sklearn.tree import DecisionTreeClassifier

model_Decs=DecisionTreeClassifier()

model_Decs.fit(X_train,Y_train.values.ravel())
Pre_Decs=model_Decs.predict(X_test)

print(classification_report(Y_test,Pre_Decs))
print(confusion_matrix(Y_test, Pre_Decs))
decs_ac=round(accuracy_score(Y_test, Pre_Decs),3)*100

decs_ac
# 3 - RandomForest





from sklearn.ensemble import RandomForestRegressor

model_ran=RandomForestRegressor()

model_ran.fit(X_train,Y_train.values.ravel())
Pre_ran=model_ran.predict(X_test)

Pre_ran=Pre_ran.astype(int)

print(classification_report(Y_test,Pre_ran))
print(confusion_matrix(Y_test, Pre_ran))

ran_ac=round(accuracy_score(Y_test, Pre_ran),3)*100

ran_ac
print("=================================================")

print("Comparing the Accuracy score for the Diff Models")

print("=================================================")

print("LogisticRegression Accuracy :", log_ac)

print("DecissionTree Accuracy :", decs_ac)

print("RandomForest Accuracy :", ran_ac)

print("=================================================")
####################################################################################################

# From the Above LogisticReggression having more Accuracy (86%). So Going with Logistic Regression #

####################################################################################################















from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(Y_test, Pre_Logi)

fpr, tpr, thresholds = roc_curve(Y_test, Pre_Logi)

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.savefig('Log_ROC')

plt.title('ROC')

plt.show()




# Now implementing the test dataset in logical regression.



test_data=pd.read_csv('../input/test.csv')

test_data_backup=test_data

test_data.head(3)
test_data.isnull().sum()
test_data['Age']=test_data.Age.fillna(test_data.Age.mean())

test_data['Fare']=test_data.Fare.fillna(test_data.Fare.mean())

test_data.isnull().sum()
test_data=test_data.drop(['Name'],axis=1)

test_data=test_data.drop(['Ticket'],axis=1)

test_data=test_data.drop(['Cabin'],axis=1)

test_data=test_data.drop(['PassengerId'],axis=1)
test_data.head()
test_data=pd.get_dummies(test_data)

test_data.head()
test_pre_Logic=model_logi.predict(test_data)
Final_Pri=test_data_backup['PassengerId']

Final_Pri=pd.DataFrame(Final_Pri)

Final_Pri['Survived']=pd.DataFrame(test_pre_Logic)

Final_Pri.head()
Final_Pri.to_csv('gender_submission.csv')