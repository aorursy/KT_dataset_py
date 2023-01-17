import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
import cufflinks as cf

cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30,color='green')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
Pclass1_mean=train[train['Pclass']==1].mean()['Age']

Pclass1_mean
Pclass2_mean=train[train['Pclass']==2].mean()['Age']

Pclass2_mean
Pclass3_mean=train[train['Pclass']==3].mean()['Age']

Pclass3_mean
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return Pclass1_mean



        elif Pclass == 2:

            return Pclass2_mean



        else:

            return Pclass3_mean



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
sns.countplot(x='Embarked',data=train,palette='RdBu_r')
train['Embarked'].fillna('S', inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
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
print (classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
evaluate = pd.read_csv('../input/test.csv')

evaluate.head()
sns.heatmap(evaluate.isnull(),yticklabels=False,cbar=False,cmap='viridis')
evaluate['Age'] = evaluate[['Age','Pclass']].apply(impute_age,axis=1)
evaluate.drop('Cabin',axis=1,inplace=True)
Pclass1_fmean=evaluate[evaluate['Pclass']==1].mean()['Fare']

Pclass1_fmean
Pclass2_fmean=evaluate[evaluate['Pclass']==2].mean()['Fare']

Pclass2_fmean
Pclass3_fmean=evaluate[evaluate['Pclass']==3].mean()['Fare']

Pclass3_fmean
def impute_fare(cols):

    Fare = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Fare):



        if Pclass == 1:

            return Pclass1_fmean



        elif Pclass == 2:

            return Pclass2_fmean



        else:

            return Pclass3_fmean



    else:

        return Fare
evaluate['Fare'] = evaluate[['Fare','Pclass']].apply(impute_fare,axis=1)
sns.heatmap(evaluate.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_result=pd.DataFrame(evaluate['PassengerId'],columns=['PassengerId'])

df_result.head()
sex = pd.get_dummies(evaluate['Sex'],drop_first=True)

embark = pd.get_dummies(evaluate['Embarked'],drop_first=True)
evaluate.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
evaluate = pd.concat([evaluate,sex,embark],axis=1)
evaluate.head()
evaluation_result = logmodel.predict(evaluate)
evaluation_result
df_result['Survived'] = pd.Series(evaluation_result, index=df_result.index)

df_result.head()
gender_submission = pd.read_csv('../input/gender_submission.csv')

gender_submission.head()
print (classification_report(gender_submission['Survived'] , df_result['Survived']))
# output data for submission in Kaggle

# df_result.to_csv('result.csv',index=False)