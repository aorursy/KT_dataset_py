import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')
total = df.append(X_test,sort=False)
total.drop('Survived',axis=1,inplace=True)
df.head()
df.info()
plt.figure(figsize=(10,6))
sns.distplot(total[total['Pclass']==1]['Age'].dropna(),label='1st class',kde=False,bins=20)
sns.distplot(total[total['Pclass']==2]['Age'].dropna(),label='2nd class',kde=False,bins=20)
sns.distplot(total[total['Pclass']==3]['Age'].dropna(),label='3rd class',kde=False,bins=20)
plt.legend()
sns.jointplot(x='Age',y='SibSp',data=total)
Corr = total[['Pclass','SibSp','Age']]
Corr = Corr.groupby(['Pclass','SibSp'],as_index=False).median()
Corr = Corr.pivot(index='Pclass',columns='SibSp',values='Age')
sns.heatmap(Corr,annot=True)
def fill_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    SibSp = cols[2]
    if pd.isnull(Age):
        return Corr[SibSp][Pclass]
    else:
        return Age
df['Age'] = df[['Age','Pclass','SibSp']].apply(fill_age,axis=1)
X_test['Age'] = X_test[['Age','Pclass','SibSp']].apply(fill_age,axis=1)
Age_Class = total[['Pclass','Age']].groupby('Pclass').median()
def fill_age2(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        return Age_Class.loc[Pclass]
    else:
        return Age
df['Age'] = df[['Age','Pclass']].apply(fill_age2,axis=1)
X_test['Age'] = X_test[['Age','Pclass']].apply(fill_age2,axis=1)
df[df['Age'].isnull()]
X_test[X_test['Age'].isnull()]
df.drop('Cabin',axis=1,inplace=True)
X_test.drop('Cabin',axis=1,inplace=True)
df.dropna(inplace=True)
Ticket = pd.DataFrame(total['Ticket'].str.split(' ',1).tolist(),columns = ['A','B'])
Ticket_string = Ticket[Ticket['B'].isnull()==False]['A']
import string
def remove_punctuations(ticket):
    nopunc = [char for char in ticket if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return nopunc.upper()
Ticket_string = Ticket_string.apply(remove_punctuations)
total['Ticket']=Ticket_string
def get_dummies(column):
    global total
    global df
    global X_test
    tmp_column = pd.get_dummies(total[column],drop_first=True)
    total.drop(column,axis=1,inplace=True)
    df.drop(column,axis=1,inplace=True)
    X_test.drop(column,axis=1,inplace=True)
    
    total  = pd.concat([total,tmp_column],axis=1)
    df     =     df.join(total.set_index('PassengerId').loc[:,str(tmp_column.columns[0]):str(tmp_column.columns[-1])])
    X_test = X_test.join(total.set_index('PassengerId').loc[:,str(tmp_column.columns[0]):str(tmp_column.columns[-1])])
def class_to_string(Pclass):
    classes = {1:'first',2:'second',3:'third'}
    return classes[Pclass]
total['Pclass'] = total['Pclass'].apply(class_to_string)
df     =     df.set_index('PassengerId')
X_test = X_test.set_index('PassengerId')
get_dummies('Ticket')
get_dummies('Sex')
get_dummies('Embarked')
get_dummies('Pclass')
df.dropna(inplace=True)
X_train = df.drop(['Survived','Name'],axis=1)
X_test.drop(['Name'],axis=1,inplace=True)
X_test.isnull().any()
X_test[X_test.isnull().T.any().T]
ThirdClassFare = X_test[X_test['third']==1]['Fare']
X_test['Fare'] = X_test['Fare'].map(lambda x: ThirdClassFare.median() if pd.isnull(x) else x)
y_train = df['Survived']
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=1000)
random_forest.fit(X_train,y_train)
predictions = random_forest.predict(X_test)
Answer=pd.DataFrame(data=X_test.index.values)
Answer['Survived'] = predictions
Answer.columns = ['PassengerId','Survived']
Answer.head()
#Answer.to_csv('../Titanic_Answer_Random_Forest.csv',index=False)