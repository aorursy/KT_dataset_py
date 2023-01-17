# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

train.head()
test=pd.read_csv("../input/test.csv")

test.head()
gender_submission=pd.read_csv("../input/gender_submission.csv")

gender_submission.head()
train.info()
test.info()
df=pd.concat([train,test],ignore_index=True)

df.info()
#identify columns with null value

df.isnull().sum()
import matplotlib.pyplot as plt

fig, axes = plt.subplots(figsize=(20,5),nrows=1,ncols=4)



df[df['Age'].isnull() & df['Cabin'].isnull()][['Fare']].hist(ax=axes[0],bins=100), axes[0].set_title('Fare Distribution with Missing Age & Cabin')

df[df['Age'].notnull() & df['Cabin'].isnull()][['Fare']].hist(ax=axes[1],bins=100),axes[1].set_title('Fare Distribution with Missing Cabin Only')

df[df['Age'].isnull() & df['Cabin'].notnull()][['Fare']].hist(ax=axes[2],bins=100),axes[2].set_title('Fare Distribution with Missing Age Only')

df[df['Embarked'].isnull()][['Fare']].hist(ax=axes[3],bins=100),axes[3].set_title('Fare Distribution with Missing Embarked Only')

df[(df['Fare'].isnull())|(df['Embarked'].isnull())]
#identify numbers of unique cabin numbers in the train dataset



cabin_df=df.groupby('Cabin').count()['PassengerId'].reset_index()

cabin_df.rename(columns={'PassengerId': 'PassengerCount'}, inplace=True)

cabin_df.info()
cabin_df.groupby('PassengerCount').count().plot.bar(figsize=(20,5))
cabin_df.groupby('PassengerCount').count()
cabin_df['CabinOccupancy']=np.where(cabin_df['PassengerCount']==1,1,'')

cabin_df['CabinOccupancy']=np.where(cabin_df['PassengerCount']==2,2,cabin_df['CabinOccupancy'])

cabin_df['CabinOccupancy']=np.where(cabin_df['PassengerCount']==3,3,cabin_df['CabinOccupancy'])

cabin_df['CabinOccupancy']=np.where(cabin_df['PassengerCount']==4,4,cabin_df['CabinOccupancy'])

cabin_df['CabinOccupancy']=np.where(cabin_df['PassengerCount']==5,5,cabin_df['CabinOccupancy'])

cabin_df['CabinOccupancy']=np.where(cabin_df['PassengerCount']==6,6,cabin_df['CabinOccupancy'])

cabin_df['CabinOccupancy']=cabin_df['CabinOccupancy'].astype(str)

cabin_df.tail()
#merge to original train dataset for further analysis

df=pd.merge(df,cabin_df[['Cabin','CabinOccupancy']],how='left',on='Cabin')

df.head()
#segragate cabin by block

block=df['Cabin'].str.split('([A-Za-z]+)(\d+)', expand=True)



#extract value 

block['block']=np.where(block[1].isnull(),block[3],block[1])

block['block']=np.where(block['block'].isnull(),block[5],block['block'])

block['block']=np.where(block['block'].isnull(),block[7],block['block'])

block['block']=np.where(block['block'].isnull(),block[9],block['block'])

block['block']=np.where(block['block'].isnull(),block[11],block['block'])



block.head()
df=pd.merge(df,block,how='outer',left_index=True,right_index=True)

df=df.drop([0,1,2,3,4,5,6,7,8,9,10,11,12], axis=1)

df.head()
#observation if correlation exist between age and fare for each block



fig, axes = plt.subplots(figsize=(20,5),nrows=2, ncols=4)



df[df['block']=='A'].plot.scatter(x='Age',y='Fare',ax=axes[0,0]);df[df['block']=='B'].plot.scatter(x='Age',y='Fare',ax=axes[0,1]);

df[df['block']=='C'].plot.scatter(x='Age',y='Fare',ax=axes[0,2]);df[df['block']=='D'].plot.scatter(x='Age',y='Fare',ax=axes[0,3]);

df[df['block']=='E'].plot.scatter(x='Age',y='Fare',ax=axes[1,0]);df[df['block']=='F'].plot.scatter(x='Age',y='Fare',ax=axes[1,1]);

df[df['block']=='G'].plot.scatter(x='Age',y='Fare',ax=axes[1,2]);
#extract titles from name

titles=df['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)

titles.rename(columns={0: 'Titles'}, inplace=True)

titles=titles.drop(columns=[1,2])

titles.head()
#merge title with df

df=pd.merge(df,titles,how='outer',left_index=True,right_index=True)

df.head()
fig, axes = plt.subplots(figsize=(20,10),nrows=2,ncols=3)



df.groupby('Titles').count()['PassengerId'].plot.barh(ax=axes[0,0],title='all passengers')

df[df['block'].isnull() & df['Age'].isnull()].groupby('Titles').count()['PassengerId'].plot.barh(ax=axes[0,1],title='missing block & age')

df[df['Age'].isnull() & df['block'].notnull()].groupby('Titles').count()['PassengerId'].plot.barh(ax=axes[0,2],title='missing age only')

df[df['block'].isnull() & df['Age'].notnull()].groupby('Titles').count()['PassengerId'].plot.barh(ax=axes[1,0],title='missing block only')

df[df['Fare'].isnull()].groupby('Titles').count()['PassengerId'].plot.barh(ax=axes[1,1],title='missing fare')

df[df['Embarked'].isnull()].groupby('Titles').count()['PassengerId'].plot.barh(ax=axes[1,2],title='missing embarked')
fig, axes = plt.subplots(figsize=(20,5),nrows=1,ncols=3)



df[df['Pclass']==1]['Fare'].hist(ax=axes[0]);axes[0].set_title('Fare Distribution for Passenger Class 1')

df[df['Pclass']==2]['Fare'].hist(ax=axes[1]);axes[1].set_title('Fare Distribution for Passenger Class 2')

df[df['Pclass']==3]['Fare'].hist(ax=axes[2]);axes[2].set_title('Fare Distribution for Passenger Class 3')
fig, axes = plt.subplots(figsize=(20,15),nrows=3,ncols=3)



df[df['block']=='A']['Fare'].hist(ax=axes[0,0]);axes[0,0].set_title('Fare Distribution for Passenger Block A')

df[df['block']=='B']['Fare'].hist(ax=axes[0,1]);axes[0,1].set_title('Fare Distribution for Passenger Block B')

df[df['block']=='C']['Fare'].hist(ax=axes[0,2]);axes[0,2].set_title('Fare Distribution for Passenger Block C')

df[df['block']=='D']['Fare'].hist(ax=axes[1,0]);axes[1,0].set_title('Fare Distribution for Passenger Block D')

df[df['block']=='E']['Fare'].hist(ax=axes[1,1]);axes[1,1].set_title('Fare Distribution for Passenger Block E')

df[df['block']=='F']['Fare'].hist(ax=axes[1,2]);axes[1,2].set_title('Fare Distribution for Passenger Block F')

df[df['block']=='G']['Fare'].hist(ax=axes[2,0]);axes[2,0].set_title('Fare Distribution for Passenger Block G')
fig, axes = plt.subplots(figsize=(20,5),nrows=1,ncols=3)



df[df['Pclass']==1].groupby('block').count()['PassengerId'].plot.barh(ax=axes[0],title='class 1')

df[df['Pclass']==2].groupby('block').count()['PassengerId'].plot.barh(ax=axes[1],title='class 2')

df[df['Pclass']==3].groupby('block').count()['PassengerId'].plot.barh(ax=axes[2],title='class 3')
fig, axes = plt.subplots(figsize=(20,5),nrows=1,ncols=3)



df[df['Pclass']==1].groupby('Embarked').count()['PassengerId'].plot.barh(ax=axes[0],title='class 1')

df[df['Pclass']==2].groupby('Embarked').count()['PassengerId'].plot.barh(ax=axes[1],title='class 2')

df[df['Pclass']==3].groupby('Embarked').count()['PassengerId'].plot.barh(ax=axes[2],title='class 3')
df.info()
#case 1a : missing fare

df[df['Fare'].isnull()].head()
#to find y_fare_test , use df with no null value for age,embarked,parck,pclass,sex,sibsp,titles as the train data

nonull_df=df[(df['Age'].notnull()) & (df['Age'].notnull()) & (df['Embarked'].notnull()) & (df['block'].notnull()) & (df['CabinOccupancy'].notnull()) ]

nonull_df.head()
nonull_df.count()['PassengerId']
#deal with categorical data for case 1a

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train=nonull_df_train[['Age','Fare','Parch','Pclass_1','Pclass_2','Pclass_3','SibSp','Embarked_S','Sex_male','Titles_ Mr']]

nonull_df_train.head()

X_nonull_df_train = nonull_df_train.drop('Fare',axis=1)

y_nonull_df_train = nonull_df_train['Fare']
#extract case 1a as test data

nullfare=df[df['Fare'].isnull()]

nullfare
nullfare=df[df['Fare'].isnull()]

cat_feats=['Embarked','Sex','block','Titles','Pclass']

nullfare_test = pd.get_dummies(nullfare,columns=cat_feats,drop_first=False)

nullfare_test=nullfare_test.drop(columns=['Name','Ticket','PassengerId','Cabin','Survived','CabinOccupancy'])

nullfare_test.head()

X_nullfare_test = nullfare_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Pclass_1','Pclass_2'])

X_nullfare_test=X_nullfare_test.drop(columns=['Fare'])

#train

from sklearn.tree import DecisionTreeRegressor

dtreeReg = DecisionTreeRegressor()

dtreeReg.fit(X_nonull_df_train,y_nonull_df_train)
#predict fare value

predictions_nullfare = dtreeReg.predict(X_nullfare_test)

predictions_nullfare
#updating data for null fare value case 1a



nullfare['Fare']=np.where(nullfare['Fare'].isnull(),93.5,nullfare['Fare'])

nullfare.head()
#updating df for case 1a



df['Fare']=np.where(df['Fare'].isnull(),93.5,df['Fare'])

df[df['PassengerId']==1044].head()
#case 1b : missing block



#deal with categorical data for case 1b

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train=nonull_df_train[['Age','Fare','Parch','SibSp',

                                 'Embarked_S','Sex_male','Titles_ Mr',

                                 'block_A','block_B','block_C','block_D',

                                'block_E','block_F','block_G','Pclass_1','Pclass_2','Pclass_3']]

nonull_df_train.head()



#multi-label classification problem
X_nonull_df_train = nonull_df_train.drop(['block_A','block_B','block_C','block_D','block_E','block_F','block_G'],axis=1)

y_nonull_df_train = nonull_df_train[['block_A','block_B','block_C','block_D','block_E','block_F','block_G']]
nullblock=nullfare #using the updated data from case 1.a for test data case 1.b

cat_feats=['Embarked','Sex','block','Titles']

nullblock_test = pd.get_dummies(nullblock,columns=cat_feats,drop_first=False)

nullblock_test=nullblock_test.drop(columns=['Name','Ticket','PassengerId','Cabin','Survived'])

nullblock_test.head()

X_nullblock_test = nullblock_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Pclass_1','Pclass_2'])

X_nullblock_test=X_nullblock_test.drop(columns=['CabinOccupancy'])

from sklearn.tree import DecisionTreeClassifier

dtreeClass = DecisionTreeClassifier()

dtreeClass.fit(X_nonull_df_train,y_nonull_df_train)
#predict cabin value

predictions_nullblock = dtreeClass.predict(X_nullblock_test)

predictions_nullblock
#updating data for null block value case 1b

nullblock['block']=np.where(nullfare['block'].isnull(),'C',nullblock['block'])

nullblock.head()
#updating df for case 1b



df['block']=np.where((df['block'].isnull()) & (df['PassengerId']==1044) ,'C',df['block'])

df[df['PassengerId']==1044].head()
#case 1c : missing cabin occupancy



#deal with categorical data for case 1c , train

cat_feats=['Embarked','Sex','Titles','block','Pclass','CabinOccupancy']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train=nonull_df_train[['Age','Fare','Parch','Pclass_1','Pclass_2','Pclass_3','SibSp',

                                 'Embarked_S','Sex_male','block_C','Titles_ Mr','CabinOccupancy_1',

                                 'CabinOccupancy_2','CabinOccupancy_3','CabinOccupancy_4','CabinOccupancy_5','CabinOccupancy_6']]

nonull_df_train.head()

X_nonull_df_train = nonull_df_train.drop(columns=['CabinOccupancy_1','CabinOccupancy_2','CabinOccupancy_3','CabinOccupancy_4','CabinOccupancy_5','CabinOccupancy_6'],axis=1)

y_nonull_df_train = nonull_df_train[['CabinOccupancy_1','CabinOccupancy_2','CabinOccupancy_3','CabinOccupancy_4','CabinOccupancy_5','CabinOccupancy_6']]
nullocc=nullfare #using the updated data from case 1.b for test data case 1.c

cat_feats=['Embarked','Sex','block','Titles','Pclass']

nullocc_test = pd.get_dummies(nullocc,columns=cat_feats,drop_first=False)

nullocc_test=nullocc_test.drop(columns=['Name','Ticket','PassengerId','Cabin','Survived'])

nullocc_test.head()

X_nullocc_test = nullocc_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Pclass_1','Pclass_2'])

X_nullocc_test=X_nullocc_test.drop(columns=['CabinOccupancy'])
dtreeClass.fit(X_nonull_df_train,y_nonull_df_train)
#predict cabin occupancy

predictions_nullocc = dtreeClass.predict(X_nullocc_test)

predictions_nullocc
#updating data for null block value case 1c

nullocc['CabinOccupancy']=np.where(nullocc['CabinOccupancy'].isnull(),'2',nullocc['CabinOccupancy'])

nullocc.head()
#updating df for case 1c



df['CabinOccupancy']=np.where((df['CabinOccupancy'].isnull()) & (df['PassengerId']==1044) ,'2',df['CabinOccupancy'])

df[df['PassengerId']==1044].head()
#updating nonull_df

nonull_df=df[(df['Fare'].notnull()) & (df['Age'].notnull()) & (df['Embarked'].notnull()) & (df['block'].notnull()) & (df['CabinOccupancy'].notnull())]

nonull_df.count()['PassengerId']
#case 2 : missing embarked

nullembark=df[df['Embarked'].isnull()]

nullembark.head()
#using the previous nonull df as the training dataset

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train = nonull_df_train[['Age','Fare','Parch','Pclass_1','Pclass_2','Pclass_3','SibSp','CabinOccupancy',

                                 'Embarked_S','Embarked_Q','Embarked_C','Sex_female','Titles_ Miss','Titles_ Mrs','block_B']]

nonull_df_train.head()
X_nonull_df_train = nonull_df_train.drop(['Embarked_S','Embarked_Q','Embarked_C'],axis=1)

y_nonull_df_train = nonull_df_train[['Embarked_S','Embarked_Q','Embarked_C']]
cat_feats=['Embarked','Sex','block','Titles','Pclass']

nullembark_test = pd.get_dummies(nullembark,columns=cat_feats,drop_first=False)

nullembark_test=nullembark_test.drop(columns=['Name','Ticket','PassengerId','Cabin','Survived'])

nullembark_test.head()

X_nullembark_test = nullembark_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Pclass_1','Pclass_2'])

#X_nullembark_test=X_nullembark_test.drop(columns=[''])
from sklearn.tree import DecisionTreeClassifier

dtreeClass = DecisionTreeClassifier()

dtreeClass.fit(X_nonull_df_train,y_nonull_df_train)
#predict embarking point

predictions_nullembark = dtreeClass.predict(X_nullembark_test)

predictions_nullembark
#updating df for case 2



df['Embarked']=np.where((df['PassengerId']==62) | (df['PassengerId']==830) & (df['Embarked'].isnull()),'S',df['Embarked'])

df[(df['PassengerId']==830) | (df['PassengerId']==62)].head()
#updating nonull_df

nonull_df=df[(df['Fare'].notnull()) & (df['Age'].notnull()) & (df['Embarked'].notnull()) & (df['block'].notnull()) ]

nonull_df.count()['PassengerId']
#case 3 : missing age only

df[df['Age'].isnull() & df['block'].notnull()].head()
#deal with categorical data for case 3

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train=nonull_df_train.drop(columns=['Cabin','Name','PassengerId','Survived','Ticket'])

nonull_df_train.head()

X_nonull_df_train = nonull_df_train.drop(['Age'],axis=1)

y_nonull_df_train = nonull_df_train[['Age']]
X_nonull_df_train.head()
nullage=df[df['Age'].isnull() & df['block'].notnull()] #extract data for case 3

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nullage_test = pd.get_dummies(nullage,columns=cat_feats,drop_first=False)

nullage_test=nullage_test.drop(columns=['Age','Name','Ticket','PassengerId','Cabin','Survived'])

nullage_test.head()

X_nullage_test = nullage_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Titles_ Capt','Titles_ Col','Titles_ Dona','Titles_ Dr','Titles_ Lady','Titles_ Major','Titles_ Master',

                                         'Titles_ Mlle','Titles_ Mme','Titles_ Sir','Titles_ the Countess','block_G'])

#X_nullage_test=X_nullage_test.drop(columns=[''])
dtreeReg.fit(X_nonull_df_train,y_nonull_df_train)
#predict age for case 3

predictions_nullage = dtreeReg.predict(X_nullage_test)

predictions_nullage

#drop exist null column for case 3 df

nullage.drop(columns=['Age'])



#merge predicted column for case  3 df

nullage['Age']=predictions_nullage



nullage.head()
#updating df for case 3

df['Age']=np.where(((df['Age'].isnull()) & (df['block'].notnull())),(df['Age'].fillna(nullage['Age'])),(df['Age']))
#updating nonull_df

nonull_df=df[(df['Fare'].notnull()) & (df['Age'].notnull()) & (df['Embarked'].notnull()) & (df['block'].notnull()) ]

nonull_df.count()['PassengerId']
#case 4 : missing block & age

df[df['block'].isnull() & df['Age'].isnull()].head()
#deal with categorical train data for case 4

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train=nonull_df_train.drop(columns=['Cabin','Name','PassengerId','Survived','Ticket'])

nonull_df_train.head()

X_nonull_df_train = nonull_df_train.drop(['Age'],axis=1)

y_nonull_df_train = nonull_df_train[['Age']]
#extract testdata for case 4.a , age

nullageblk=df[df['block'].isnull() & df['Age'].isnull()] 

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nullageblk_test = pd.get_dummies(nullageblk,columns=cat_feats,drop_first=False)

nullageblk_test=nullageblk_test.drop(columns=['Age','Name','Ticket','PassengerId','Cabin','Survived'])

nullageblk_test.head()

X_nullageblk_test = nullageblk_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Titles_ Capt','Titles_ Col','Titles_ Dona','Titles_ Lady','Titles_ Major',

                                         'Titles_ Mlle','Titles_ Mme','Titles_ Sir','Titles_ the Countess','block_G',

                                         'block_A','block_B','block_C','block_D','block_E','block_F','CabinOccupancy'])

X_nullageblk_test=X_nullageblk_test.drop(columns=['CabinOccupancy','Titles_ Ms'])
dtreeReg.fit(X_nonull_df_train,y_nonull_df_train)
#predict age for case 4.a

predictions_nullageblk = dtreeReg.predict(X_nullageblk_test)

predictions_nullageblk
#drop exist null column for case 4.a df

nullageblk.drop(columns=['Age'])



#merge predicted column for case  4.a df

nullageblk['Age']=predictions_nullageblk



nullageblk
#deal with categorical train data for case 4.b, cabin occupancy

cat_feats=['Embarked','Sex','Titles','block','Pclass','CabinOccupancy']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train=nonull_df_train.drop(columns=['Cabin','Name','PassengerId','Survived','Ticket'])

nonull_df_train.head()

X_nonull_df_train = nonull_df_train.drop(columns=['CabinOccupancy_1','CabinOccupancy_2','CabinOccupancy_3','CabinOccupancy_4','CabinOccupancy_5','CabinOccupancy_6'],axis=1)

y_nonull_df_train = nonull_df_train[['CabinOccupancy_1','CabinOccupancy_2','CabinOccupancy_3','CabinOccupancy_4','CabinOccupancy_5','CabinOccupancy_6']]
#extract test data for case 4.b , cabin occ

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nullageblk_test = pd.get_dummies(nullageblk,columns=cat_feats,drop_first=False)

nullageblk_test=nullageblk_test.drop(columns=['Name','Ticket','PassengerId','Cabin','Survived'])

nullageblk_test.head()

X_nullageblk_test = nullageblk_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Titles_ Capt','Titles_ Col','Titles_ Dona','Titles_ Lady','Titles_ Major',

                                         'Titles_ Mlle','Titles_ Mme','Titles_ Sir','Titles_ the Countess','block_G',

                                         'block_A','block_B','block_C','block_D','block_E','block_F'])

X_nullageblk_test=X_nullageblk_test.drop(columns=['Titles_ Ms','CabinOccupancy'])
dtreeClass.fit(X_nonull_df_train,y_nonull_df_train)
#predict age for case 4.b

predictions_nullageblk = dtreeClass.predict(X_nullageblk_test)

predictions_nullageblk
predictions_nullageblk=pd.DataFrame(predictions_nullageblk, columns=['1','2','3','4','5','6'])

predictions_nullageblk['CabinOccupancy']=np.where(predictions_nullageblk['1']==1,'1','')

predictions_nullageblk['CabinOccupancy']=np.where(predictions_nullageblk['2']==1,'2',predictions_nullageblk['CabinOccupancy'])

predictions_nullageblk['CabinOccupancy']=np.where(predictions_nullageblk['3']==1,'3',predictions_nullageblk['CabinOccupancy'])

predictions_nullageblk['CabinOccupancy']=np.where(predictions_nullageblk['4']==1,'4',predictions_nullageblk['CabinOccupancy'])

predictions_nullageblk['CabinOccupancy']=np.where(predictions_nullageblk['5']==1,'5',predictions_nullageblk['CabinOccupancy'])

predictions_nullageblk['CabinOccupancy']=np.where(predictions_nullageblk['6']==1,'6',predictions_nullageblk['CabinOccupancy'])

predictions_nullageblk.head()
predictions_nullageblk=predictions_nullageblk['CabinOccupancy'].values
#drop exist null column for case 4.b df

nullageblk.drop(columns=['CabinOccupancy'])



#merge predicted column for case  4.b df

nullageblk['CabinOccupancy']=predictions_nullageblk



nullageblk
#deal with categorical data for case 4.c

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train=nonull_df_train.drop(columns=['Cabin','Name','PassengerId','Survived','Ticket'])

nonull_df_train.head()

X_nonull_df_train = nonull_df_train.drop(['block_A','block_B','block_C','block_D','block_E','block_F','block_G'],axis=1)

y_nonull_df_train = nonull_df_train[['block_A','block_B','block_C','block_D','block_E','block_F','block_G']]
#extract test data for case 4.c , cabin block

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nullageblk_test = pd.get_dummies(nullageblk,columns=cat_feats,drop_first=False)

nullageblk_test=nullageblk_test.drop(columns=['Name','Ticket','PassengerId','Cabin','Survived'])

nullageblk_test.head()

X_nullageblk_test=nullageblk_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Titles_ Capt','Titles_ Col','Titles_ Dona','Titles_ Lady','Titles_ Major',

                                         'Titles_ Mlle','Titles_ Mme','Titles_ Sir','Titles_ the Countess'])



X_nullageblk_test=X_nullageblk_test.drop(columns=['Titles_ Ms'])
dtreeClass.fit(X_nonull_df_train,y_nonull_df_train)
#predict block for case 4.c

predictions_nullageblk = dtreeClass.predict(X_nullageblk_test)

predictions_nullageblk
predictions_nullageblk=pd.DataFrame(predictions_nullageblk, columns=['block_A','block_B','block_C','block_D','block_E','block_F','block_G'])

predictions_nullageblk['block']=np.where(predictions_nullageblk['block_A']==1,'A','')

predictions_nullageblk['block']=np.where(predictions_nullageblk['block_B']==1,'B',predictions_nullageblk['block'])

predictions_nullageblk['block']=np.where(predictions_nullageblk['block_C']==1,'C',predictions_nullageblk['block'])

predictions_nullageblk['block']=np.where(predictions_nullageblk['block_D']==1,'D',predictions_nullageblk['block'])

predictions_nullageblk['block']=np.where(predictions_nullageblk['block_E']==1,'E',predictions_nullageblk['block'])

predictions_nullageblk['block']=np.where(predictions_nullageblk['block_F']==1,'F',predictions_nullageblk['block'])

predictions_nullageblk['block']=np.where(predictions_nullageblk['block_G']==1,'G',predictions_nullageblk['block'])



predictions_nullageblk.head()
predictions_nullageblk=predictions_nullageblk['block'].values
#drop exist null column for case 4.c df

nullageblk.drop(columns=['block'])



#merge predicted column for case  4.c df

nullageblk['block']=predictions_nullageblk



nullageblk
#updating df for case 4 by filling empty field for age, cabin occupancy and block using predicted value for 4a, 4b & 4c



df['Age'] = df['Age'].mask(df['Age'].eq(0)).fillna(df['PassengerId'].map(nullageblk.set_index('PassengerId')['Age']))

df['CabinOccupancy'] = df['CabinOccupancy'].mask(df['CabinOccupancy'].eq(0)).fillna(df['PassengerId'].map(nullageblk.set_index('PassengerId')['CabinOccupancy']))

df['block'] = df['block'].mask(df['block'].eq(0)).fillna(df['PassengerId'].map(nullageblk.set_index('PassengerId')['block']))

#check 1 sample if updated

df[df['PassengerId']==6]
#updating nonull_df

nonull_df=df[(df['Fare'].notnull()) & (df['Age'].notnull()) & (df['Embarked'].notnull()) & (df['block'].notnull()) ]

nonull_df.count()['PassengerId']

#case 5: missing block only

df[df['block'].isnull() & df['Age'].notnull()].head()
#deal with categorical data for case 5.a , CabinOccupancy

cat_feats=['Embarked','Sex','Titles','block','Pclass','CabinOccupancy']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train=nonull_df_train.drop(columns=['Cabin','Name','PassengerId','Survived','Ticket'])

nonull_df_train.head()

X_nonull_df_train = nonull_df_train.drop(columns=['CabinOccupancy_1','CabinOccupancy_2','CabinOccupancy_3','CabinOccupancy_4','CabinOccupancy_5','CabinOccupancy_6'],axis=1)

y_nonull_df_train = nonull_df_train[['CabinOccupancy_1','CabinOccupancy_2','CabinOccupancy_3','CabinOccupancy_4','CabinOccupancy_5','CabinOccupancy_6']]
#extract test data for case 5.a 

nullblk=df[df['block'].isnull() & df['Age'].notnull()]

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nullblk_test = pd.get_dummies(nullblk,columns=cat_feats,drop_first=False)

nullblk_test=nullblk_test.drop(columns=['Name','Ticket','PassengerId','Cabin','Survived'])

nullblk_test.head()

X_nullblk_test=nullblk_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Titles_ Capt','Titles_ Dona','Titles_ Lady','Titles_ Major',

                                             'Titles_ Mlle','Titles_ Mme','Titles_ Sir','Titles_ the Countess',

                                             'block_A','block_B','block_C','block_D','block_E','block_F','block_G'])



X_nullblk_test=X_nullblk_test.drop(columns=['CabinOccupancy','Titles_ Jonkheer','Titles_ Don','Titles_ Rev'])
X_nonull_df_train.head(1)
X_nullblk_test.head(1)
dtreeClass.fit(X_nonull_df_train,y_nonull_df_train)
#predict occupancy for case 5a

predictions_nullblk = dtreeClass.predict(X_nullblk_test)

predictions_nullblk
predictions_nullblk=pd.DataFrame(predictions_nullblk, columns=['1','2','3','4','5','6'])

predictions_nullblk['CabinOccupancy']=np.where(predictions_nullblk['1']==1,'1','')

predictions_nullblk['CabinOccupancy']=np.where(predictions_nullblk['2']==1,'2',predictions_nullblk['CabinOccupancy'])

predictions_nullblk['CabinOccupancy']=np.where(predictions_nullblk['3']==1,'3',predictions_nullblk['CabinOccupancy'])

predictions_nullblk['CabinOccupancy']=np.where(predictions_nullblk['4']==1,'4',predictions_nullblk['CabinOccupancy'])

predictions_nullblk['CabinOccupancy']=np.where(predictions_nullblk['5']==1,'5',predictions_nullblk['CabinOccupancy'])

predictions_nullblk['CabinOccupancy']=np.where(predictions_nullblk['6']==1,'6',predictions_nullblk['CabinOccupancy'])



predictions_nullblk.head()
predictions_nullblk=predictions_nullblk['CabinOccupancy'].values
#drop exist null column for case 5.a 

nullblk.drop(columns=['CabinOccupancy'])



#merge predicted column for case  5.a

nullblk['CabinOccupancy']=predictions_nullblk



nullblk
#deal with categorical train data for case 5.b

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nonull_df_train = pd.get_dummies(nonull_df,columns=cat_feats,drop_first=False)

nonull_df_train=nonull_df_train.drop(columns=['Cabin','Name','PassengerId','Survived','Ticket'])

nonull_df_train.head()

X_nonull_df_train = nonull_df_train.drop(['block_A','block_B','block_C','block_D','block_E','block_F','block_G'],axis=1)

y_nonull_df_train = nonull_df_train[['block_A','block_B','block_C','block_D','block_E','block_F','block_G']]
#test data for case 5b

cat_feats=['Embarked','Sex','Titles','block','Pclass']

nullblk_test = pd.get_dummies(nullblk,columns=cat_feats,drop_first=False)

nullblk_test=nullblk_test.drop(columns=['Name','Ticket','PassengerId','Cabin','Survived'])

nullblk_test.head()

X_nullblk_test=nullblk_test
#drop X train and X test feature to standardize matrix size



X_nonull_df_train=X_nonull_df_train.drop(columns=['Titles_ Capt','Titles_ Dona','Titles_ Lady','Titles_ Major',

                                             'Titles_ Mlle','Titles_ Mme','Titles_ Sir','Titles_ the Countess'])



X_nullblk_test=X_nullblk_test.drop(columns=['Titles_ Jonkheer','Titles_ Don','Titles_ Rev'])
dtreeClass.fit(X_nonull_df_train,y_nonull_df_train)
#predict block for case 5.b

predictions_nullblk = dtreeClass.predict(X_nullblk_test)

predictions_nullblk
predictions_nullblk=pd.DataFrame(predictions_nullblk, columns=['block_A','block_B','block_C','block_D','block_E','block_F','block_G'])

predictions_nullblk['block']=np.where(predictions_nullblk['block_A']==1,'A','')

predictions_nullblk['block']=np.where(predictions_nullblk['block_B']==1,'B',predictions_nullblk['block'])

predictions_nullblk['block']=np.where(predictions_nullblk['block_C']==1,'C',predictions_nullblk['block'])

predictions_nullblk['block']=np.where(predictions_nullblk['block_D']==1,'D',predictions_nullblk['block'])

predictions_nullblk['block']=np.where(predictions_nullblk['block_E']==1,'E',predictions_nullblk['block'])

predictions_nullblk['block']=np.where(predictions_nullblk['block_F']==1,'F',predictions_nullblk['block'])

predictions_nullblk['block']=np.where(predictions_nullblk['block_G']==1,'G',predictions_nullblk['block'])



predictions_nullblk.head()
predictions_nullblk=predictions_nullblk['block'].values
#drop exist null column for case 5b 

nullblk.drop(columns=['block'])



#merge predicted column for case  5b

nullblk['block']=predictions_nullblk



nullblk
#updating df for case 5 by filling empty field for cabin occupancy and block using predicted value for 5a, 5b



df['CabinOccupancy'] = df['CabinOccupancy'].mask(df['CabinOccupancy'].eq(0)).fillna(df['PassengerId'].map(nullblk.set_index('PassengerId')['CabinOccupancy']))

df['block'] = df['block'].mask(df['block'].eq(0)).fillna(df['PassengerId'].map(nullblk.set_index('PassengerId')['block']))

#check 1 sample if updated

df[df['PassengerId']==9]
#check df for nan value on important fields



df.info()
train.info()
#updating empty train dataset for age, cabin occupancy and block using df



train['Age'] = train['Age'].mask(train['Age'].eq(0)).fillna(train['PassengerId'].map(df.set_index('PassengerId')['Age']))

train['Embarked'] = train['Embarked'].mask(train['Embarked'].eq(0)).fillna(train['PassengerId'].map(df.set_index('PassengerId')['Embarked']))



train=pd.merge(train,df[['PassengerId','CabinOccupancy','block','Titles']],how='left',on='PassengerId')
train.info()
#observed survived and association with other category , also use Cramer's for numerical indication



fig, axes = plt.subplots(figsize=(20,15),nrows=2,ncols=3)



train.groupby(['Survived','Sex']).size().unstack().plot(ax=axes[0,0],kind='bar', stacked=True)

train.groupby(['Survived','Pclass']).size().unstack().plot(ax=axes[0,1],kind='bar', stacked=True)

train.groupby(['Survived','block']).size().unstack().plot(ax=axes[0,2],kind='bar', stacked=True)

train.groupby(['Survived','Embarked']).size().unstack().plot(ax=axes[1,0],kind='bar', stacked=True)

train.groupby(['Survived','Titles']).size().unstack().plot(ax=axes[1,1],kind='bar', stacked=True)

train.groupby(['Survived','CabinOccupancy']).size().unstack().plot(ax=axes[1,2],kind='bar', stacked=True)
import scipy

import scipy.stats as ss #for cramer's v function on categorical correlation

from scipy.stats import pearsonr #categorical correlation

from scipy import stats #categorical - numerical correlation Point Biserial

#source function from https://stackoverflow.com/questions/46498455/categorical-features-correlation

def cramers_v(confusion_matrix):

    """ calculate Cramers V statistic for categorial-categorial association.

        uses correction from Bergsma and Wicher,

        Journal of the Korean Statistical Society 42 (2013): 323-328

    """

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum()

    phi2 = chi2 / n

    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
confusion_matrix = pd.crosstab(train["Survived"], train["Sex"]).as_matrix()

cramers_v(confusion_matrix)
scipy.stats.spearmanr(train["Survived"], train["Sex"]) #compare result with spearman method
confusion_matrix = pd.crosstab(train["Survived"], train["Pclass"]).as_matrix()

cramers_v(confusion_matrix)
scipy.stats.spearmanr(train["Survived"], train["Pclass"]) #compare result with spearman method
confusion_matrix = pd.crosstab(train["Survived"], train["block"]).as_matrix()

cramers_v(confusion_matrix)
confusion_matrix = pd.crosstab(train["Survived"], train["Embarked"]).as_matrix()

cramers_v(confusion_matrix)
confusion_matrix = pd.crosstab(train["Survived"], train["Titles"]).as_matrix()

cramers_v(confusion_matrix)
confusion_matrix = pd.crosstab(train["Survived"], train["CabinOccupancy"]).as_matrix()

cramers_v(confusion_matrix)
confusion_matrix = pd.crosstab(train["Survived"], train["SibSp"]).as_matrix()

cramers_v(confusion_matrix)
confusion_matrix = pd.crosstab(train["Survived"], train["Parch"]).as_matrix()

cramers_v(confusion_matrix)
scipy.stats.pointbiserialr(train["Survived"], train["Age"]) #correlation between categorical and conti value
scipy.stats.pointbiserialr(train["Survived"], train["Fare"]) #correlation between categorical and conti value
#drop features not require for ML

train_=train.drop(columns=['PassengerId','Name','Ticket','Cabin',

                           'Fare','Age','Parch','SibSp','CabinOccupancy',

                          'Embarked'])

train_.info()
cat_feats=['Sex','block','Titles','Pclass']
#deal with categorical train data

train_ = pd.get_dummies(train_,columns=cat_feats,drop_first=False)
test.info()
#updating empty test dataset for age, cabin occupancy and block using df



test['Age'] = test['Age'].mask(test['Age'].eq(0)).fillna(test['PassengerId'].map(df.set_index('PassengerId')['Age']))

test['Fare'] = test['Fare'].mask(test['Fare'].eq(0)).fillna(test['PassengerId'].map(df.set_index('PassengerId')['Fare']))



test=pd.merge(test,df[['PassengerId','CabinOccupancy','block','Titles']],how='left',on='PassengerId')

test.info()
#drop features not require for ML

test_=test.drop(columns=['PassengerId','Name','Ticket','Cabin',

                           'Fare','Age','Parch','SibSp','CabinOccupancy',

                          'Embarked'])

test_.info()
#deal with categorical test data

test_ = pd.get_dummies(test_,columns=cat_feats,drop_first=False)
X_train = train_.drop('Survived',axis=1)

y_train = train_['Survived']

X_train.head(1)
X_test=test_
X_test.head(1)
#drop X train and X test feature to standardize matrix size



X_train=X_train.drop(columns=['Titles_ the Countess','Titles_ Sir','Titles_ Mme',

                              'Titles_ Mlle','Titles_ Major','Titles_ Lady','Titles_ Jonkheer',

                             'Titles_ Don','Titles_ Capt'])



X_test=X_test.drop(columns=['Titles_ Dona'])
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
predictions_dtree = dtree.predict(X_test)
predictions_dtree
#load y_test to observed accuracy

y_test=gender_submission['Survived'].values
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions_dtree))
print(confusion_matrix(y_test,predictions_dtree))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)

rfc.fit(X_train,y_train)
predictions_rfc = rfc.predict(X_test)
predictions_rfc
print(classification_report(y_test,predictions_rfc))
print(confusion_matrix(y_test,predictions_rfc))
#merge predicted column in test data

test['Survived']=predictions_rfc



test.head()
predictions=test[['PassengerId','Survived']]
predictions.to_csv('output.csv', index=False)