import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sea

sea.set_style('whitegrid')
from sklearn.preprocessing import LabelEncoder,PowerTransformer
df_train=pd.read_csv("../input/titanic/train.csv")

df_test=pd.read_csv("../input/titanic/test.csv")

ID_test=df_test['PassengerId']
df_train.head()
sea.countplot(df_train['SibSp'])
df_train['SibSp'].value_counts()
df_train.sort_values(by=['SibSp'],ascending=False).head(10)
outliner_SibSp=df_train.loc[df_train['SibSp']==8]

outliner_SibSp
df_train=df_train.drop(outliner_SibSp.index,axis=0)
df_train.loc[df_train['SibSp']==8]
sea.boxplot(df_train['Fare'],orient='v')
df_train.sort_values(by=['Fare','Pclass'],ascending=False).head(10)
outliner_Fare=df_train.loc[df_train['Fare']>500]

outliner_Fare
df_train=df_train.drop(outliner_Fare.index,axis=0)
df_train.shape
df_test.shape
dataset=pd.concat([df_train,df_test],ignore_index=True)
dataset.head()
dataset.shape
dataset=dataset.fillna(np.nan)

dataset.isnull().sum()
dataset.loc[dataset['Embarked'].isnull()]
sea.countplot(dataset['Embarked'])
dataset['Embarked']=dataset['Embarked'].fillna('S')
dataset.loc[dataset['Fare'].isnull()]
dataset.loc[(dataset['Pclass']==3)].sort_values(by=['Fare'],ascending=False).head(15)
temp=dataset.loc[(dataset['Pclass']==3) & (dataset['Parch']==0) & (dataset['SibSp']==0) & (dataset['Fare']>0)].sort_values(by=['Fare'],ascending=False)

temp.head()
dataset['Fare']=dataset['Fare'].fillna(temp['Fare'].mean())
g= sea.FacetGrid(df_train,col='Survived')

g= g.map(sea.distplot,'Age')
nullAgeSubset=dataset.loc[dataset['Age'].isnull()]

nullAgeSubset.shape
for index in nullAgeSubset.index:

    ageSubsetMean=dataset['Age'].loc[(dataset['Parch']==nullAgeSubset.loc[index]['Parch'])&(dataset['SibSp']==nullAgeSubset.loc[index]['SibSp'])&(dataset['Pclass']==nullAgeSubset.loc[index]['Pclass'])&(dataset['Embarked']==nullAgeSubset.loc[index]['Embarked'])&(dataset['Sex']==nullAgeSubset.loc[index]['Sex'])].mean()

    if(ageSubsetMean>0):

        dataset['Age'].loc[index]=ageSubsetMean

    else:

        ageSubsetMean=dataset['Age'].loc[(dataset['Parch']==nullAgeSubset.loc[index]['Parch'])&(dataset['SibSp']==nullAgeSubset.loc[index]['SibSp'])&(dataset['Pclass']==nullAgeSubset.loc[index]['Pclass'])&(dataset['Embarked']==nullAgeSubset.loc[index]['Embarked'])].mean()

        if(ageSubsetMean>0):

            dataset['Age'].loc[index]=ageSubsetMean

        else:

            ageSubsetMean=dataset['Age'].loc[(dataset['Parch']==nullAgeSubset.loc[index]['Parch'])&(dataset['SibSp']==nullAgeSubset.loc[index]['SibSp'])&(dataset['Pclass']==nullAgeSubset.loc[index]['Pclass'])].mean()

            if(ageSubsetMean>0):

                dataset['Age'].loc[index]=ageSubsetMean

            else:

                dataset['Age'].loc[index]=dataset['Age'].mean()

                
dataset['Age'].isnull().sum()
sea.heatmap(df_train.corr(),cmap='BrBG',annot=True)
sea.countplot(dataset['Sex'],hue=dataset['Survived'])
sea.catplot(data=dataset,x='Pclass',y='Survived',kind='bar')
g=sea.FacetGrid(data=dataset.loc[dataset['Survived']==1],col='Pclass')

g=g.map(sea.countplot,'Sex')
dataset.head()
sea.distplot(np.array(dataset['Fare']).reshape(-1,1),axlabel='Fare')
sea.distplot(PowerTransformer().fit_transform(np.array(dataset['Fare']).reshape(-1,1)),axlabel='Fare')
sea.distplot(np.array(dataset['Age']).reshape(-1,1),axlabel='Age')
sea.distplot(PowerTransformer().fit_transform(np.array(dataset['Age']).reshape(-1,1)),axlabel='Age')
sea.distplot(np.array(dataset['SibSp']).reshape(-1,1),axlabel='SibSp')
sea.distplot(PowerTransformer().fit_transform(np.array(dataset['SibSp']).reshape(-1,1)),axlabel='SibSp')
sea.distplot(np.array(dataset['Parch']).reshape(-1,1),axlabel='Parch')
sea.distplot(PowerTransformer().fit_transform(np.array(dataset['Parch']).reshape(-1,1)),axlabel='Parch')
X=dataset.drop(['Cabin','Name','PassengerId','Survived','Ticket'],axis=1)

Y=dataset['Survived']
X.head(10)
X['Age']=PowerTransformer().fit_transform(np.array(X['Age']).reshape(-1,1))

X['Fare']=PowerTransformer().fit_transform(np.array(X['Fare']).reshape(-1,1))

X['Parch']=PowerTransformer().fit_transform(np.array(X['Parch']).reshape(-1,1))

X['Sex']=LabelEncoder().fit_transform(X['Sex'])

X['SibSp']=PowerTransformer().fit_transform(np.array(X['SibSp']).reshape(-1,1))

dummyPclass=pd.get_dummies(X['Pclass'],prefix='Pclass')

dummyEmbarked=pd.get_dummies(X['Embarked'],prefix='Embarked')

X=pd.concat([X.drop(['Pclass','Embarked'],axis=1),dummyPclass,dummyEmbarked],axis=1)
X.head(15)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from xgboost import XGBClassifier
X_pro=PolynomialFeatures(degree=2).fit_transform(X)
trainDataX=X_pro[:df_train.shape[0]]

trainDataY=Y[:df_train.shape[0]].astype('int32')

testDataX=X_pro[df_train.shape[0]:]
X_train,X_test,Y_train,Y_test=train_test_split(trainDataX,trainDataY,test_size=0.1,random_state=47)
model=XGBClassifier(learning_rate=0.001,n_estimators=300,max_depth=30)

#model=SVC(kernel='poly',C=100,gamma=0.1)

model.fit(X_train,Y_train)

accuracy_score(Y_train,model.predict(X_train))
accuracy_score(Y_test,model.predict(X_test))
submission=pd.DataFrame(columns=['PassengerId','Survived'])

submission['PassengerId']=ID_test

submission['Survived']=model.predict(testDataX)
submission.head()
filename='submission.csv'

submission.to_csv(filename,index=False)

from IPython.display import FileLink

FileLink(filename)