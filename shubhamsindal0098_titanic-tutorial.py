import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
print("The shape of train dataset is {}".format(train_data.shape))
print("The shape of test dataset is {}".format(test_data.shape))
print(train_data.info())
print('-'*40)
print(test_data.info())
print(train_data.isnull().sum())
print('-'*40)
print(test_data.isnull().sum())
import missingno as msno
msno.bar(train_data)
msno.bar(test_data)
train_data.columns
cols=['Survived','Sex','Pclass','SibSp','Parch','Embarked']
for i in cols:
    print(train_data[i].value_counts())
cols=['Survived','Sex','Pclass','SibSp','Parch','Embarked']
n_rows=2
n_cols=3
fig,axs=plt.subplots(n_rows,n_cols,figsize=(n_cols*3.2,n_rows*3.2))
for r in range(0,n_rows):
    for c in range(0,n_cols):
        i=r*n_cols + c
        ax=axs[r][c]
        sns.countplot(train_data[cols[i]],hue=train_data['Survived'],ax=ax)
        ax.set_title(cols[i])
        ax.legend(title='Survived',loc='upper right')
plt.tight_layout()
dataset=[train_data,test_data]
for val in dataset:
    val['FamilySize']=val['SibSp']+val['Parch']+1
train_data=train_data.drop(['SibSp','Parch','PassengerId','Name','Cabin'],axis=1)
test_data=test_data.drop(['SibSp','Parch','PassengerId','Name','Cabin'],axis=1)
train_data.head()
g=sns.catplot(x='Survived',hue='Pclass',data=train_data.sort_values(by='Pclass',ascending=False),kind='count')
g.set(xticklabels=['Died','Survived'])

g=sns.catplot(x='Survived',hue='Pclass',col='Sex',data=train_data.sort_values(by='Pclass',ascending=False),kind='count')
g.set(xticklabels=['Died','Survived'])
g=sns.catplot(x='Survived',hue='Pclass',row='Embarked',col='Sex',data=train_data.sort_values(by='Pclass',ascending=False),kind='count')
g.set(xticklabels=['Died','Survived'])
train_data['Embarked']=train_data['Embarked'].fillna(train_data['Embarked'].mode())
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].median())

train_data=train_data.dropna()
test_data=test_data.dropna()
sns.heatmap(train_data.isnull(),yticklabels=False)
sns.heatmap(test_data.isnull(),yticklabels=False)
train_data['AgeBand']=pd.cut(train_data['Age'],4)
test_data['AgeBand']=pd.cut(test_data['Age'],4)
train_data.head()
train_data[['Survived','AgeBand']].groupby('AgeBand',as_index=False).sum().sort_values(by='AgeBand',ascending=True)
dataset=[train_data,test_data]
for val in dataset:
    val.loc[(val['Age'] <= 20.315),'Age']=0
    val.loc[(val['Age'] > 20.315) & (val['Age'] <= 40.21), 'Age']=1
    val.loc[(val['Age'] > 40.21) & (val['Age'] <= 60.105), 'Age']=2
    val.loc[(val['Age'] > 60.105) & (val['Age'] <= 80.0) ,'Age']=3
train_data.head()
train_data['Age'].value_counts()
train_data['FareBand']=pd.qcut(train_data['Fare'],5)
test_data['FareBand']=pd.qcut(test_data['Fare'],5)
train_data[['Survived','FareBand']].groupby('FareBand',as_index=False).sum().sort_values(by='FareBand',ascending=True)
for val in dataset:
    val.loc[(val['Fare'] <= 7.902),'Fare']=0
    val.loc[(val['Fare'] > 7.902) & (val['Fare'] <= 12.925), 'Fare']=1
    val.loc[(val['Fare'] > 12.925) & (val['Fare'] <= 26.0), 'Fare']=2
    val.loc[(val['Fare'] > 26.0) & (val['Fare'] <= 46.9) ,'Fare']=3
    val.loc[(val['Fare'] > 46.9) & (val['Fare'] <= 512.329) ,'Fare']=4
train_data.head()
test_data.head()
train_data=train_data.drop(['Ticket','AgeBand','FareBand'],axis=1)
test_data=test_data.drop(['Ticket','AgeBand','FareBand'],axis=1)
for val in dataset:
    val['Sex']=val['Sex'].map({'female':0,'male':1})
train_data['Embarked'].value_counts().to_dict()
for val in dataset:
    val['Embarked']=val['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
train_data.head()
