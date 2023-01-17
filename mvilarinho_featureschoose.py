import pandas as pd
exemplo=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
data=pd.read_csv('/kaggle/input/titanic/train.csv',index_col='PassengerId')
dataTest=pd.read_csv('/kaggle/input/titanic/test.csv',index_col='PassengerId')
data=data.dropna(axis=0)
data.head()
dataS=data.loc[data.Survived==1]
rowsS,_=dataS.shape
rowsT,_=data.shape
dataS.groupby('Embarked')['Survived'].count()/rowsS,data.groupby('Embarked')['Survived'].count()/rowsT
dataS.groupby('SibSp')['Survived'].count()/rowsS,data.groupby('SibSp')['Survived'].count()/rowsT
dataS.groupby('Parch')['Survived'].count()/rowsS,data.groupby('Parch')['Survived'].count()/rowsT
data['Cabin2']=data.Cabin.map(lambda x: x[0])
dataS=data.loc[data.Survived==1]
dataS.groupby('Cabin2')['Survived'].count()/rowsS,data.groupby('Cabin2')['Survived'].count()/rowsT