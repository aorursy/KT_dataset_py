#Import all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style("darkgrid")

#load the train data
train_data= pd.read_csv("../input/titanic/train.csv")
test_data=pd.read_csv("../input/titanic/test.csv")
train_data.head()

train_data.info()
train_data.describe(include="all") # for statistical information which is not for categorical data
# Lets check for other missing data
train_data.isna().sum()
sns.barplot(x='Sex',y='Survived',data=train_data)
plt.show();
sns.barplot(x='Pclass',y='Survived',data=train_data)
plt.show();
sns.barplot(x='SibSp',y='Survived',data=train_data)
plt.show();
sns.barplot(x='Parch',y='Survived',data=train_data)
plt.show();
# Age feature
#sort the ages into logical categories


bins = [ 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = [ 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train_data['AgeGroup'] = pd.cut(train_data["Age"], bins, labels = labels)
test_data['AgeGroup'] = pd.cut(test_data["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.set()
plt.figure(figsize = (12,6))
sns.barplot(x="AgeGroup", y="Survived", data=train_data,palette='magma')
plt.show()
#First we will have a loot at test data also
test_data.describe(include='all')
# We will start with dropping 'Cabin' column, as a lot of data is missing
train_data.drop("Cabin",axis=1,inplace=True)
test_data.drop("Cabin",axis=1,inplace=True)
# Embarked feature
print(f'Number of people living in Southampton are(S){train_data[train_data["Embarked"]=="S"].shape[0]}')
print(f'Number of people living in Cherbourg are(S){train_data[train_data["Embarked"]=="C"].shape[0]}')
print(f'Number of people living in Queenstown are(S){train_data[train_data["Embarked"]=="Q"].shape[0]}')
# As we can see most of the passengers live in Southampton, so we will the missing data with 'S'
train_data.fillna({"Embarked":"S"},inplace=True)
#train_data[train_data.isnull().any(axis = 1)] # to find the rows having missing data, it will return empty df
# We will drop the name column, as it has no much of significance
train_data.drop("Name",axis=1,inplace=True)
test_data.drop("Name",axis=1,inplace=True)


#We will drop the row in test data for 'Fare' column
test_data['Fare']=test_data["Fare"].fillna(test_data["Fare"].mean())
# Fill the missing value of Age column with mean value.
train_data['Age'] = train_data['Age'].fillna(train_data.groupby('Sex')['Age'].transform('mean'))

test_data['Age'] = test_data['Age'].fillna(test_data.groupby('Sex')['Age'].transform('mean'))
test_data.isna().sum()
#Map the categorical colum 'Sex' with numerical data
sex_mapping = {"male": 0, "female": 1}
train_data['Sex'] = train_data['Sex'].map(sex_mapping)
test_data['Sex'] = test_data['Sex'].map(sex_mapping)

train_data.head()
#Mapping for Embarked feature
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)

train_data.head()
# Drop the AgeGroup column as it was created for visualization purpose
train_data.drop("AgeGroup",axis=1,inplace=True)

test_data.drop("AgeGroup",axis=1,inplace=True)
# drop the ticket column
train_data.drop("Ticket",axis=1,inplace=True)
test_data.drop("Ticket",axis=1,inplace=True)
train_data.head()
test_data.head()
X= train_data.drop(["PassengerId","Survived"],axis=1) # Our samples
y= train_data["Survived"] # Our targets

X.shape,y.shape
#Lets split the data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

from sklearn.svm import SVC

svc=SVC()
svc.fit(X_train,y_train)
svc_preds=svc.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,svc_preds)
#As accuracy is not good, we can try Random Forest
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)

rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_preds= rfc.predict(X_test)

accuracy_score(y_test,rfc_preds)
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
gbk_preds = gbk.predict(X_test)
accuracy_score(y_test,gbk_preds)
ids = test_data['PassengerId']
predictions = rfc.predict(test_data.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission_2.csv', index=False)