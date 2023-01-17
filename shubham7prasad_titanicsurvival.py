# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
trainData=pd.read_csv('/kaggle/input/titanic/train.csv')

testData=pd.read_csv('/kaggle/input/titanic/test.csv')

trainData.head()
trainData.info()
testData.head()
import seaborn as sns



sns.barplot(x=trainData.Sex,y=trainData['Survived'])
#Data cleansing

missing_value_count_coulmn=trainData.isnull().sum()

print(missing_value_count_coulmn[missing_value_count_coulmn>0])

#missingValues=pd.DataFrame(data=missing_value_count_coulmn[missing_value_count_coulmn>0],columns=['A','B'])

missingValues=pd.DataFrame(missing_value_count_coulmn[missing_value_count_coulmn>0])

list(missingValues.index)

trainData.shape
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

sns.distplot(a=trainData['Age'],kde=False)

#sns.distplot(a=trainData['Fare'],kde=False)
sns.countplot(x='Sex',data=trainData)

sns.countplot(x='Sex',hue='Survived',data=trainData)


#trainData['flag']=0

#testData['flag']=1

#data=pd.concat([trainData,testData])

data=trainData

data.groupby(['Sex','Survived'])['Survived'].count()

data.head()
data[['Sex','Survived']].groupby(['Sex']).mean().sort_values(by='Survived')
data[['Pclass','Survived']].groupby(['Pclass']).mean().sort_values(by='Survived')
sns.countplot(x='Pclass',hue='Survived',data=data)
pd.crosstab(data.Pclass,data.Survived)
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True)
sns.factorplot('Pclass','Survived',hue='Sex',data=data)
plt.figure(figsize=(16,10))

sns.violinplot('Pclass','Age',hue='Survived',data=data,split=True)

#sns.facetgrid(a,col="time")

sns.violinplot('Sex','Age',hue='Survived',data=data,split=True)
sns.countplot(x='Embarked',hue='Survived',data=data)
data.groupby(['Embarked','Survived'])['Survived'].count()
data[['Embarked','Survived']].groupby(['Embarked']).mean().sort_values(by='Survived')
sns.violinplot('Embarked','Age',hue='Survived',data=data,split=True)
sns.heatmap(data.corr(),annot=True)
df_all_corr = data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'Age']
print(missing_value_count_coulmn[missing_value_count_coulmn>0])
data['Age']=data.groupby(['Sex','Pclass'])['Age'].apply(lambda x:x.fillna(x.median()))

print(data.isnull().sum())
from sklearn.impute import SimpleImputer



imputer=SimpleImputer(strategy='most_frequent')

nullCategoryValues=['Embarked']

dataAgeimputed=pd.DataFrame(imputer.fit_transform(data[nullCategoryValues]))

dataAgeimputed.columns=data[nullCategoryValues].columns

dataAgeimputed.head()

data['Embarked']=dataAgeimputed['Embarked']





print(data.isnull().sum())

data['Initial']=0

for i in data:

    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')

    

data['Initial'].head()

print(data['Initial'].unique())
data['Age_band']=0

data.loc[data['Age']<=16,'Age_band']=0

data.loc[(data['Age']>16) & (data['Age']<=32),'Age_band']=1

data.loc[(data['Age']>32) & (data['Age']<=48),'Age_band']=2

data.loc[(data['Age']>48) & (data['Age']<=64),'Age_band']=3

data.loc[(data['Age']>64),'Age_band']=4



data.head()
data['Age_band'].value_counts()
sns.factorplot('Age_band','Survived',data=data,col='Pclass')
data['Family_Size']=0

data['Family_Size']=data['Parch']+data['SibSp']

data['Alone']=0

data.loc[data['Family_Size']==0,'Alone']=1



sns.factorplot('Family_Size','Survived',data=data,col='Pclass')
sns.factorplot('Alone','Survived',data=data)
data['Fare_Range']=pd.qcut(data['Fare'],4)



data.groupby(['Fare_Range'])['Fare_Range'].count()
data.groupby(['Fare_Range'])['Survived'].mean()
data['Fare_cat']=0

data.loc[data['Fare']<=7.91,'Fare_cat']=0

data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1

data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2

data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3

sns.factorplot('Fare_Range','Survived',data=data,hue='Sex')
data['Sex'].replace(['male','female'],[0,1],inplace=True)



data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)



data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)



data.head()

#data['Initial'].unique()

data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)

data.head()
from sklearn.ensemble import RandomForestRegressor



model=RandomForestRegressor(random_state=1)





testData['Sex'].replace(['male','female'],[0,1],inplace=True)



testData['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

testData['Fare_cat']=0

testData.loc[testData['Fare']<=7.91,'Fare_cat']=0

testData.loc[(testData['Fare']>7.91)&(testData['Fare']<=14.454),'Fare_cat']=1

testData.loc[(testData['Fare']>14.454)&(testData['Fare']<=31),'Fare_cat']=2

testData.loc[(testData['Fare']>31)&(testData['Fare']<=513),'Fare_cat']=3



testData.head()
testData['Family_Size']=0

testData['Family_Size']=testData['Parch']+testData['SibSp']

testData['Alone']=0

testData.loc[testData['Family_Size']==0,'Alone']=1

testData.head()



testData['Age_band']=0

testData.loc[testData['Age']<=16,'Age_band']=0

testData.loc[(testData['Age']>16) & (testData['Age']<=32),'Age_band']=1

testData.loc[(testData['Age']>32) & (testData['Age']<=48),'Age_band']=2

testData.loc[(testData['Age']>48) & (testData['Age']<=64),'Age_band']=3

testData.loc[(testData['Age']>64),'Age_band']=4



testData.head()
testData['Initial']=0

for i in testData:

    testData['Initial']=testData.Name.str.extract('([A-Za-z]+)\.')

    

testData['Initial'].head()

print(testData['Initial'].unique())

testData['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

testData['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)



testData.head()
testData.drop(['Name','Age','Ticket','Fare','Cabin','PassengerId'],axis=1,inplace=True)

testData.head()
#model.fit(data,testData)

Y=data['Survived']

X=data.drop(['Survived'],axis=1,inplace=True)

data.head()



model.fit(data,Y)
test_Y=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
melb_preds = model.predict(data)

#print(mean_absolute_error(test_Y, melb_preds))