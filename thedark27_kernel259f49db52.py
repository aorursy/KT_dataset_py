import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data=pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv')
data.head()
data.isnull()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=data)
import cufflinks as cf

cf.go_offline()

data['Fare'].iplot(kind='hist',bins=50)
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age

    
data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.drop('Cabin',axis=1,inplace=True)
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.drop(['Sex','Embarked','Name','Ticket'],inplace=True,axis=1)
data.head()
x=data.drop('Survived',axis=1)
y=data['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.30,random_state=101)
from sklearn.tree import DecisionTreeClassifier
logmodel=DecisionTreeClassifier()
logmodel.fit(X_train,y_train)
prediction=logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))
from sklearn.ensemble import RandomForestClassifier
result=RandomForestClassifier()
result.fit(X_train,y_train)
result_pred=result.predict(X_test)
print(confusion_matrix(y_test,result_pred))
print(classification_report(y_test,result_pred))
submission = pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':result_pred})



#Visualize the first 5 rows

submission.head()
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)