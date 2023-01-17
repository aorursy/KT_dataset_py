import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
titanic_train=pd.read_csv('../input/train.csv')
titanic_train.head(5)
sns.heatmap(titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=titanic_train)
plt.figure(figsize=(10,6))
sns.boxplot('Pclass','Age',data=titanic_train)
def impute_age(cols):
    Age=cols[0]
    pclass = cols[1]
    
    if pd.isnull(Age):
        
        if pclass == 1:
            return 37
        elif pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
titanic_train['Age']=titanic_train[['Age','Pclass']].apply(impute_age,axis=1)
titanic_train.drop('Cabin',axis=1,inplace=True)
titanic_train.fillna('C',inplace=True)
sns.heatmap(titanic_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex=pd.get_dummies(titanic_train['Sex'],drop_first=True)
sex.head()
embark=pd.get_dummies(titanic_train['Embarked'],drop_first=True)
embark.head()
titanic_train = pd.concat([titanic_train,sex,embark],axis=1)
titanic_train.head()
titanic_train.drop(['Sex','Embarked','Ticket','Name'],axis=1,inplace=True)
titanic_train.head()
titanic_train.drop('PassengerId',axis=1,inplace=True)
titanic_train.head()
pclass=pd.get_dummies(titanic_train['Pclass'],drop_first=True)
titanic_train=pd.concat([titanic_train,pclass],axis=1)
titanic_train.head()
titanic_train.drop('Pclass',axis=1,inplace=True)
test=pd.read_csv('../input/test.csv')
test.head()
sns.heatmap(data=test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test['Age']=test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
test.fillna(0,inplace=True)
test.head()
# test.dropna(inplace=True)
embark=pd.get_dummies(test['Embarked'],drop_first=True)
embark.head()
sex=pd.get_dummies(test['Sex'],drop_first=True)
pclass=pd.get_dummies(test['Pclass'],drop_first=True)
test=pd.concat([test,sex,embark,pclass],axis=1)
test.head()
test.drop(['Pclass','Name','Sex','Ticket'],axis=1,inplace=True)
test.head()
test1=test.drop('PassengerId',axis=1).copy()
test1.drop('Embarked',axis=1,inplace=True)
test.drop('Embarked',axis=1,inplace=True)
titanic_train.head()
test1.head()
X=titanic_train.drop('Survived',axis=1)
y=titanic_train['Survived']
X.head()
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X,y)
prediction=logmodel.predict(test1)
prediction.shape
submission=pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived' : prediction
})
submission.to_csv('submission.csv',index=False)
submission=pd.read_csv('submission.csv')
submission.head()
