import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv('../input/train.csv')
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(data=train,x='Survived',hue='Pclass')
sns.distplot(train['Age'].dropna(),kde=False,bins=30)



sns.boxplot(x='Pclass',y='Age',data=train)
def calculate_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
train[['Age','Pclass']].head()
train['Age']=train[['Age','Pclass']].apply(calculate_age,axis=1)
sns.heatmap(train.isnull(),cmap='viridis',yticklabels=False,cbar=False)
train.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train.isnull(),cmap='viridis',yticklabels=False,cbar=False)
train.dropna(inplace=True)
sns.heatmap(train.isnull(),cmap='viridis',yticklabels=False,cbar=False)
train.head()
pd.get_dummies(train['Sex']).head(2)
sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)
train=pd.concat([train,sex,embark],axis=1)
train.head(2)

train.drop('PassengerId',axis=1,inplace=True)
train.head()
PC=pd.get_dummies(train['Pclass'],drop_first=True)
train=pd.concat([train,PC],axis=1)
train.head()
#train.drop([2,3],axis=1,inplace=True)
train.drop(['Embarked'],axis=1,inplace=True)
train.drop(['Name','Sex','Ticket','Pclass'],axis=1,inplace=True)
X=train.drop('Survived',axis=1)
y=train['Survived']

X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
test_data = pd.read_csv('../input/test.csv')
test_data.head()
sns.boxplot(x=test_data['Pclass'],y=test_data['Age'],data=test_data)
test_data['Age']=test_data[['Age','Pclass']].apply(calculate_age,axis=1)
test_data.head()
sns.heatmap(test_data.isnull(),cmap='viridis',yticklabels=False,cbar=False)
test_data.columns
test_data.info()

sns.boxplot(x='Pclass',y='Fare',data=test_data)
test_data[test_data['Pclass'] == 3]['Fare'].median()
def calculate_Fare(cols):
    Fare=cols[0]
    Pclass=cols[1]
    if pd.isnull(Fare):
        if Pclass == 1:
            return 60
        elif Pclass==2:
            return 15.75
        else:
            return 7.8958
    else:
        return Fare
test_data['Fare'] = test_data[['Fare','Pclass']].apply(calculate_Fare,axis=1)


sex = pd.get_dummies(test_data['Sex'],drop_first=True)
embark = pd.get_dummies(test_data['Embarked'],drop_first=True)
PC = pd.get_dummies(test_data['Pclass'],drop_first=True)
test_data = pd.concat([test_data,sex,embark,PC],axis=1)
test_data.head()
test_data.drop(['Sex','Embarked','Pclass','Name','Ticket','Cabin'],axis=1,inplace=True)
test_data.head(2)

test_data.info()


idx.head()
idx.rename(columns={'idx':'PassengerId'},inplace = True)
idx.head()
test_data.drop('PassengerId',inplace=True,axis=1)
predictions_new = logmodel.predict(test_data)

survived = pd.DataFrame({'Survived':predictions_new})
survived.head()
submission=pd.concat([idx,survived],axis=1)
submission.head()
OUTPUT_RESULT="submission1.csv"
submission.to_csv(OUTPUT_RESULT,index=False)
