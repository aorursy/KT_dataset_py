import os
print(os.listdir("../input"))
train_filepath='../input/train.csv'
test_filepath='../input/test.csv'
import numpy as np
import pandas as pd
df_train0 = pd.read_csv(train_filepath,decimal=",")
df_test0 = pd.read_csv(test_filepath,decimal=",")
df_train0.head()
df_test0.head()
df_train0['Survived'].value_counts(normalize=True)

df_test0['Survived']=-1
df_train0.columns
df_test0=df_test0.reindex(columns=df_train0.columns)
df_test0.head()
df_all=pd.concat([df_train0,df_test0]).copy()
df_all.head()
df_all.dtypes
df_all.isnull().sum()
df_all['Age'].describe()
dg=df_all[['Sex','Pclass','Age']].dropna().copy()
dg.head()
dg['Age']=pd.to_numeric(dg['Age'])
import seaborn as sns
sns.boxplot(x='Pclass', y='Age', data=dg)
sns.boxplot(x='Sex', y='Age', data=dg)
def input_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1: return 38
        elif Pclass==2: return 30
        else: return 25
    else: return Age
df_all['Age']=df_all[['Age','Pclass']].apply(input_age,axis=1)
df_all['Age']=df_all['Age'].astype(float)
df_all['Fare'].describe()
bool_val_not_null=np.invert(df_all['Fare'].isnull())
fares_num=pd.to_numeric(df_all['Fare'][bool_val_not_null])
np.average(fares_num)
df_all.loc[df_all['Fare'].isnull(),'Fare']=33.3
df_all['Fare']=df_all['Fare'].astype(float)
df_all['Embarked'].value_counts(normalize=True)
df_all.loc[df_all['Embarked'].isnull(),'Embarked']='S'
df_all.head(10)
df_all['Cabin'].unique()
df_all.loc[df_all['Cabin'].isnull(),'Cabin']='XX'
df_all.isnull().sum()
df_all['Cabin']=df_all['Cabin'].str.extract('(.)')
df_all.head()
n=df_all['Name'].str.extract('.+?,\s(.+?).\s')[0]
n.value_counts()
df_all[n=='Jonkheer']
df_all[n=='th']
df_all[n=='Master'].head()
n[n=='Jonkheer']='Nh'
n[n=='Sir']='Nh'
n[n=='th']='Nh'
n[n=='Mlle']='Miss'
n[n=='Mme']='Mrs'
n[n=='Lady']='Miss'
n[n=='Dona']='Miss'
n[n=='Don']='Rev'
n[n=='Ms']='Miss'
n[n=='Major']='Nh'
n[n=='Capt']='Nh'
n[n=='Col']='Nh'
n.value_counts()
df_all['Title']=n.copy()
df_all['Parch'].value_counts()
df_all['Fam']=df_all['SibSp']+df_all['Parch']
df_all.head()
df=df_all.copy()
df.head()
df.to_csv('titanic_full.csv',index=False)
df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.drop('SibSp', axis=1, inplace=True)
df.drop('Parch', axis=1, inplace=True)
df.head()
df.to_csv('titanic_clean.csv',index=False)
sex=pd.get_dummies(df['Sex'],drop_first=True)
embarked=pd.get_dummies(df['Embarked'],drop_first=True)
title=pd.get_dummies(df['Title'],drop_first=True)
cabin=pd.get_dummies(df['Cabin'],drop_first=True)
df=pd.concat([df,sex,embarked,title,cabin], axis=1)
df.drop('Embarked', axis=1, inplace=True)
df.drop('Sex', axis=1, inplace=True)
df.drop('Title', axis=1, inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df.head()
train=df[df['Survived']>=0].copy()
test=df[df['Survived']==-1].copy()
test=test.drop('Survived',axis=1)
train.head()
test.head()
y = train['Survived']
X = train.drop('Survived',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)
from sklearn.linear_model import LogisticRegression
lrm = LogisticRegression()
lrm.fit(X_train,y_train)
y_test_pred=lrm.predict(X_test)
from sklearn.metrics import confusion_matrix
C = confusion_matrix(y_test,y_test_pred)
print(C)
accuracy=(C[0,0]+C[1,1])/(C[0,0]+C[1,0]+C[0,1]+C[1,1])
print(accuracy)
lrm.fit(X,y)
y_pred=lrm.predict(test)
subm=pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
subm.head()
subm.to_csv('titanic_third.csv', index=False)