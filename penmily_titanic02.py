# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from pandas import Series,DataFrame



data_train=pd.read_csv('../input/train.csv')

data_train
data_train.info()
data_train.describe()
import matplotlib.pyplot as plt

fig=plt.figure()

fig.set(alpha=0.2)



plt.subplot2grid((2,3),(0,0))

data_train.Survived.value_counts().plot(kind='bar')

plt.title(u"survived or not(survived:1)")

plt.ylabel(u"quantities")



plt.subplot2grid((2,3),(0,1))

data_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u'quantities')

plt.title(u'social class distribution')



plt.subplot2grid((2,3),(0,2))

plt.scatter(data_train.Survived,data_train.Age)

plt.ylabel(u'age')

plt.grid(b=True,which='major',axis='y')

plt.title(u'survived or not by age(survived:1)')



plt.subplot2grid((2,3),(1,0),colspan=2)

data_train.Age[data_train.Pclass==1].plot(kind='kde')

data_train.Age[data_train.Pclass==2].plot(kind='kde')

data_train.Age[data_train.Pclass==3].plot(kind='kde')

plt.xlabel(u'age')

plt.ylabel(u'people density')

plt.title(u'social class by age distribution')

plt.legend((u'1st',u'2nd',u'3rd'),loc='best')



plt.subplot2grid((2,3),(1,2))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title(u'onboard quantities on each entrance')

plt.ylabel(u'quantities')

plt.show()
fig.set(alpha=0.2)



Survived_0=data_train.Pclass[data_train.Survived==0].value_counts()

Survived_1=data_train.Pclass[data_train.Survived==1].value_counts()

df=pd.DataFrame({u'live':Survived_1,u'died':Survived_0})

df.plot(kind='bar',stacked=True)

plt.title(u'live or not by social class')

plt.xlabel(u'social class')

plt.ylabel(u'quantities')

plt.show()
fig.set(alpha=0.2)



Survived_m=data_train.Survived[data_train.Sex=='male'].value_counts()

Survived_f=data_train.Survived[data_train.Sex=='female'].value_counts()

df=pd.DataFrame({u'man':Survived_m,u'woman':Survived_f})

df.plot(kind='bar',stacked=True)

plt.title(u'live or not by sex')

plt.xlabel(u'sex')

plt.ylabel(u'quantities')

plt.show()
fig=plt.figure()

fig.set(alpha=0.65)

plt.title(u'survive or not by pclass & sex')



ax1=fig.add_subplot(141)

data_train.Survived[data_train.Sex=='female'][data_train.Pclass!=3].value_counts().plot(kind='bar',label='female highclass',color='#FA2479')

ax1.set_xticklabels([u'live',u'died'],rotation=0)

ax1.legend([u'woman/highclass'],loc='best')



ax2=fig.add_subplot(142,sharey=ax1)

data_train.Survived[data_train.Sex=='female'][data_train.Pclass==3].value_counts().plot(kind='bar',label='female lowclass',color='pink')

ax2.set_xticklabels([u'died',u'live'],rotation=0)

plt.legend([u'woman/lowclass'],loc='best')



ax3=fig.add_subplot(143,sharey=ax1)

data_train.Survived[data_train.Sex=='male'][data_train.Pclass!=3].value_counts().plot(kind='bar',label='male highclass',color='lightblue')

ax3.set_xticklabels([u'died',u'live'],rotation=0)

plt.legend([u'man/highclass'],loc='best')



ax4=fig.add_subplot(144,sharey=ax1)

data_train.Survived[data_train.Sex=='male'][data_train.Pclass==3].value_counts().plot(kind='bar',label='male lowclass',color='blue')

ax4.set_xticklabels([u'died',u'live'],rotation=0)

plt.legend([u'man/lowclass'],loc='best')



plt.show()
fig=plt.figure()

fig.set(alpha=0.2)



Survived_0=data_train.Embarked[data_train.Survived==0].value_counts()

Survived_1=data_train.Embarked[data_train.Survived==1].value_counts()

df=pd.DataFrame({u'lived':Survived_1,u'died':Survived_0})

df.plot(kind='bar',stacked=True)

plt.title(u'live or not by onboard entrance')

plt.xlabel(u'onboard entrance')

plt.ylabel(u'quantities')



plt.show()
fig=plt.figure()

fig.set(alpha=0.2)



Survived_cabin=data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()

Survived_nocabin=data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()

df=pd.DataFrame({u'have':Survived_cabin,u'no':Survived_nocabin}).transpose()

df.plot(kind='bar',stacked=True)

plt.title(u'live or not by cabin')

plt.xlabel(u'cabin')

plt.ylabel(u'quantities')

plt.show()
from sklearn.ensemble import RandomForestRegressor



def set_missing_ages(df):

    age_df=df[['Age','Fare','Parch','SibSp','Pclass']]

    known_age=age_df[age_df.Age.notnull()].as_matrix()

    unknown_age=age_df[age_df.Age.isnull()].as_matrix()

    y=known_age[:,0]

    X=known_age[:,1:]

    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)

    rfr.fit(X,y)

    predictedAges=rfr.predict(unknown_age[:,1::])

    df.loc[(df.Age.isnull()),'Age']=predictedAges

    return (df,rfr)



def set_Cabin_type(df):

    df.loc[(df.Cabin.notnull()),'Cabin']='Yes'

    df.loc[(df.Cabin.isnull()),'Cabin']='No'

    return (df)

data_train,rfr=set_missing_ages(data_train)

data_train=set_Cabin_type(data_train)
data_train.info()