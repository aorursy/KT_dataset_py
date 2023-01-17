import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline 
train=pd.read_csv(r'/kaggle/input/titanic/train.csv')
test=pd.read_csv(r'/kaggle/input/titanic/test.csv')
train.tail()
test.head()
len(train)
len(test)
train.drop(['Cabin','Name','Ticket'],inplace=True,axis=1)
test.drop(['Cabin','Name','Ticket'],inplace=True,axis=1)
train.head(3)
test.head(3)
train['Embarked']=pd.get_dummies(train['Embarked'],drop_first=True)
test['Embarked']=pd.get_dummies(test['Embarked'],drop_first=True)
#train.drop(['Embarked'],inplace=True,axis=1)
#test.drop(['Embarked'],inplace=True,axis=1)
train.head(3)
test.head(4)
train['Sex']=pd.get_dummies(train['Sex'],drop_first=True)
train.head(3)
train.head()
test['Sex']=pd.get_dummies(test['Sex'],drop_first=True)
test.head(3)
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())
train[train['Age'].isnull()==True]
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute(col):

    Age=col[0]

    Pclass=col[1]

    

    if pd.isnull(Age)==True:

        

        if Pclass==1:

            return 37

        

        elif Pclass==2:

            return 29

        else:

            return 24

        

    else:

        return Age
train['Age']=train[['Age','Pclass']].apply(impute,axis=1)
sns.heatmap(train.isnull())
sns.boxplot(x='Pclass',y='Age',data=test,palette='winter')
def impute2(col):

    Age=col[0]

    Pclass=col[1]

    

    if pd.isnull(Age)==True:

        

        if Pclass==1:

            return 42

        

        elif Pclass==2:

            return 26

        else:

            return 25

        

    else:

        return Age
test['Age']=test[['Age','Pclass']].apply(impute2,axis=1)
sns.heatmap(test.isnull())
len(test)
test['Fare'].plot(kind='kde')
test['Fare'].mean()
test['Fare']=test['Fare'].fillna(35.62)
sns.heatmap(test.isnull())
len(test)
X_train=train.drop('Survived',axis=1)

X_train
Y_train=train['Survived']

Y_train
X_test=test
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=300)
clf.fit(X_train,Y_train)
prediction=clf.predict(X_test)
#from sklearn.tree import DecisionTreeRegressor
#clf=DecisionTreeRegressor(max_depth=4)

#clf=clf.fit(X_train,Y_train)
#prediction=clf.predict(X_test)
#from sklearn.linear_model import LogisticRegression 
#logmodel=LogisticRegression(solver='lbfgs',max_iter=500)

#logmodel.fit(X_train,Y_train)
#predictions=logmodel.predict(X_test)
pred=pd.DataFrame(data=prediction,columns=['Survived'])

pred
#def num(a):

 #   if a >= 0.655:

  #      return 1

   # else:

    #    return 0
#pred=pred['Survived'].apply(num)

#print(pred)
ps_id=test['PassengerId']
final_df=pd.concat([ps_id,pred],axis=1)

final_df
final_df['Survived'].unique()
final_df.to_csv('resultX122.csv', index=True,encoding='utf-8')