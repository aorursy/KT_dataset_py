import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train = pd.read_csv("../input/train.csv")
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,
            cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x="Survived",hue="Pclass",data=train)
sns.distplot(train['Age'].dropna(),kde=False,bins=30)
train.info()
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(bins=40,figsize=(12,6))
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y="Age",data=train)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
test = pd.read_csv("../input/test.csv")
test.head()
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
test.dropna(inplace=True)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex,embark],axis=1)
test.head()
predictions = logmodel.predict(test)
output = pd.Series(predictions)
output.to_csv("output.csv")
