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

import matplotlib.pyplot as plt

import seaborn as sns



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('max.rows',900)



df=pd.read_csv('/kaggle/input/titanic/train.csv')

df.set_index('PassengerId',inplace=True)



test=pd.read_csv('/kaggle/input/titanic/test.csv')

test.set_index('PassengerId',inplace=True)
list_of_numerics=df.select_dtypes(include=['float','int']).columns

types= df.dtypes

missing= round((df.isnull().sum()/df.shape[0]),3)*100

overview= df.apply(lambda x: [round(x.min()),round(x.max()),round(x.mean()),round(x.quantile(0.5))] if x.name in list_of_numerics else x.unique())





outliers= df.apply(lambda x: sum(

                                 (x<(x.quantile(0.25)-1.5*(x.quantile(0.75)-x.quantile(0.25))))|

                                 (x>(x.quantile(0.75)+1.5*(x.quantile(0.75)-x.quantile(0.25))))

                                 if x.name in list_of_numerics else ''))





explo = pd.DataFrame({'Types': types,

                      'Missing%': missing,

                      'Overview': overview,

                      'Outliers': outliers}).sort_values(by=['Missing%','Types'],ascending=False)

explo.transpose()
df.drop(['Cabin','Ticket'],axis=1,inplace=True)

test.drop(['Cabin','Ticket'],axis=1,inplace=True)
values=df.groupby(['Sex', 'Pclass'])['Age'].agg(['mean', 'median']).round(1)

values
filt=(df['Sex']=='female')&(df['Pclass']==1)&(df['Age'].isnull())

df.loc[filt,'Age']=35.0



filt=(df['Sex']=='female')&(df['Pclass']==2)&(df['Age'].isnull())

df.loc[filt,'Age']=28.0



filt=(df['Sex']=='female')&(df['Pclass']==3)&(df['Age'].isnull())

df.loc[filt,'Age']=22.0



filt=(df['Sex']=='male')&(df['Pclass']==1)&(df['Age'].isnull())

df.loc[filt,'Age']=40.0



filt=(df['Sex']=='male')&(df['Pclass']==2)&(df['Age'].isnull())

df.loc[filt,'Age']=30.0



filt=(df['Sex']=='male')&(df['Pclass']==3)&(df['Age'].isnull())

df.loc[filt,'Age']=25.0



df['Embarked'].replace({np.nan:'S'},inplace=True) ##(mode of df['Embarked'] comes out to be S, can be found out by df['Embarked'].mode())

values=test.groupby(['Sex', 'Pclass'])['Age'].agg(['mean', 'median']).round(1)

values
filt=(test['Sex']=='female')&(test['Pclass']==1)&(test['Age'].isnull())

test.loc[filt,'Age']=41.0



filt=(test['Sex']=='female')&(test['Pclass']==2)&(test['Age'].isnull())

test.loc[filt,'Age']=24.0



filt=(test['Sex']=='female')&(test['Pclass']==3)&(test['Age'].isnull())

test.loc[filt,'Age']=22.0



filt=(test['Sex']=='male')&(test['Pclass']==1)&(test['Age'].isnull())

test.loc[filt,'Age']=42.0



filt=(test['Sex']=='male')&(test['Pclass']==2)&(test['Age'].isnull())

test.loc[filt,'Age']=28.0



filt=(test['Sex']=='male')&(test['Pclass']==3)&(test['Age'].isnull())

test.loc[filt,'Age']=24.0



test['Fare'].replace({np.nan:test['Fare'].median()},inplace=True)

ticket_fare=test.groupby(['Pclass'])['Fare'].agg(['mean', 'median']).round(1)

ticket_fare
filt=(test['Pclass']==1)&(test['Fare'].isnull())

test.loc[filt,'Fare']=94.3



filt=(test['Pclass']==2)&(test['Fare'].isnull())

test.loc[filt,'Fare']=22.2



filt=(test['Pclass']==3)&(test['Fare'].isnull())

test.loc[filt,'Fare']=12.5
fig, axarr = plt.subplots(1,2,figsize=(12,6))

a = sns.countplot(df['SibSp'], ax=axarr[0]).set_title('Passengers count by SibSp')



b = sns.barplot(x='SibSp', y='Survived', data=df, ax=axarr[1]).set_ylabel('Survival rate')

axarr[1].set_title('Survival rate by SibSp')
fig, axarr = plt.subplots(1,2,figsize=(12,6))

a = sns.countplot(df['Parch'], ax=axarr[0]).set_title('Passengers count by Parch')

axarr[1].set_title('Survival rate by Parch')

b = sns.barplot(x='Parch', y='Survived', data=df, ax=axarr[1]).set_ylabel('Survival rate')
plt.figure(figsize=(20,20))

sns.heatmap(data=df.corr(),xticklabels=True,yticklabels=True,cbar=True,linecolor='white',annot=True)
df['Fam_size'] = df['SibSp'] + df['Parch'] + 1

test['Fam_size'] = test['SibSp'] + test['Parch'] + 1



df['Fam_type'] = pd.cut(df.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

test['Fam_type'] = pd.cut(test.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

test['Title'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())



df['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)

test['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)



df['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)

test['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)



##drop the features as we have created new features in place

df.drop(['SibSp','Parch','Fam_size','Name'],axis=1,inplace=True)

test.drop(['SibSp','Parch','Fam_size','Name'],axis=1,inplace=True)



df.head()
df=pd.get_dummies(df)

test=pd.get_dummies(test)



"""

Seperating the train data into X and y



"""

X=df.drop('Survived',axis=1)

y=df['Survived']
from sklearn.ensemble import GradientBoostingClassifier



clf=GradientBoostingClassifier(random_state=0,max_features='auto')





clf.fit(xtrain,ytrain)

probabilities=clf.predict_proba(X)

importance=list(zip(X.columns,clf.feature_importances_))



print('accuracy'+' '+'='+' '+str(clf.score(X,y)

))

print()

print("Feature Importances:-")

print()

print(importance)
from sklearn.metrics import roc_curve



fpr1,tpr1,thresholds1=roc_curve(y,probabilities[:,1])



plt.figure(figsize=(8,8))

sns.lineplot(y=tpr1,x=fpr1,ci=None)
pred=clf1.predict(test)

Id=test.index

final_sub=pd.Series(data=pred,index=Id,name='Survived')

final_sub.to_csv('final_sub1.csv')

final_sub.head()