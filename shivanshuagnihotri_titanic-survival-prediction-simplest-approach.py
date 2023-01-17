import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
train_data=pd.read_csv('../input/titanic/train.csv')
train_data.head()

test_data=pd.read_csv('../input/titanic/train.csv')
test_data.head()


train_data.describe()
sns.heatmap(train_data.isnull(),yticklabels=False) # check missing values
sns.barplot(x="Sex", y="Survived", data=train_data)
print("males survived:", train_data["Survived"][train_data["Sex"] == 'male'].value_counts(normalize=True))
sns.barplot(x="Pclass", y="Survived", data=train_data) #survived by class
sns.barplot(x="Pclass", y="Survived", data=train_data,hue='Sex')
# dropping missing values coulumn
train_data = train_data.drop(['Cabin'], axis = 1)
test_data = test_data.drop(['Cabin'], axis = 1)
# cleaning and processing data 
sns.boxplot(x='Pclass',y='Age',data=train_data)
# first class people have higher age


# age column has missing values 
#applying function to set age according to class 
# for example class 1 people might be having average age 40 

def set_age(cols):
    age=cols[0]
    pclass=cols[1]
    
    if pd.isnull(age):
        
        if pclass==1:
            return 40
        elif pclass == 2:
            return 30
        else:
            return 25
    
    else: 
        return age
    
    
train_data['Age']=train_data[['Age','Pclass']].apply(set_age,axis=1)
test_data['Age']=test_data[['Age','Pclass']].apply(set_age,axis=1)


sns.heatmap(train_data.isnull(),yticklabels=False)  #no missing values now
train_sex_new=pd.get_dummies(train_data['Sex'],drop_first=True) #replacing categorcial data with numerical value
train_embark_new=pd.get_dummies(train_data['Embarked'],drop_first=True)

test_sex_new=pd.get_dummies(test_data['Sex'],drop_first=True) #replacing categorcial data with numerical value
test_embark_new=pd.get_dummies(test_data['Embarked'],drop_first=True)
train_data=pd.concat([train_data,train_sex_new,train_embark_new],axis=1)
test_data=pd.concat([test_data,test_sex_new,test_embark_new],axis=1)
train_data.drop(['Sex','Embarked'],axis=1,inplace=True)
test_data.drop(['Sex','Embarked'],axis=1,inplace=True)


train_data.head()


# dropping unnecessary columns 

train_data = train_data.drop(['Ticket','Name'], axis = 1)
test_data = test_data.drop(['Ticket','Name'], axis = 1)
train_data.head() #all numerical values
test_data.head()  # all numericals
# final check on missing data
print(pd.isnull(train_data).sum())
print(pd.isnull(test_data).sum()) #one fare value missing in test data
# filling missing fare value based on mean fare value for respective clas s
for i in range(len(test_data["Fare"])):
    if pd.isnull(test_data["Fare"][i]):
        pc = test_data["Pclass"][i] # respective Pclass 
        test_data["Fare"][i] = round(train_data[train_data["Pclass"] == pc]["Fare"].mean(), 4)
print(pd.isnull(test_data).sum())
# train test split 
from sklearn.model_selection import train_test_split

X = train_data.drop(['Survived', 'PassengerId'], axis=1)
y = train_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))

from sklearn.metrics import accuracy_score
a = round(accuracy_score(y_pred, y_test) * 100, 2)
print(a) 
 # 79 percent accuracy
# Support Vector Machines

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
a = round(accuracy_score(y_pred, y_test) * 100, 2)
print(a)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
a = round(accuracy_score(y_pred, y_test) * 100, 2)
print(a)
'''
To submit to kaggle first predict on test_data given by kaggle as: 

# predicting on given test data

#set ids as PassengerId and prediction 
pid = test_data['PassengerId']
predictions = rf.predict(test_data.drop('PassengerId', axis=1))

#sconvert to csv file named submission.csv

submit = pd.DataFrame({ 'PassengerId' : pid, 'Survived': predictions })
submit.to_csv('titanic_submission2.csv', index=False)


'''