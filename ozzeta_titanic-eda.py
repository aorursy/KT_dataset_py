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
from matplotlib import pyplot
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
# Sex

Sex_surv = pd.crosstab(train['Sex'],train['Survived'],normalize='index')

print(Sex_surv)
# Sex グラフ

height1=Sex_surv.iloc[:,1]

height2=Sex_surv.iloc[:,0]

bar1=pyplot.bar(Sex_surv.index,height1)

bar2=pyplot.bar(Sex_surv.index,height2,bottom=height1,color="red")

pyplot.title("Sex-Survive Pct")

pyplot.legend((bar1[0], bar2[0]), ("survived", "dead"))

pyplot.xlabel("Sex")
# Pclass

Pclass_surv = pd.crosstab(train['Pclass'],train['Survived'],normalize='index')

print(Pclass_surv)
# Pclass グラフ

height1=Pclass_surv.iloc[:,1]

height2=Pclass_surv.iloc[:,0]

bar1=pyplot.bar(Pclass_surv.index,height1)

bar2=pyplot.bar(Pclass_surv.index,height2,bottom=height1,color="red")

pyplot.title("Pclass-Survive Pct")

pyplot.legend((bar1[0], bar2[0]), ("survived", "dead"))

pyplot.xticks(np.arange(1,len(Pclass_surv)+1))

pyplot.xlabel("Pclass")
# SibSp 同乗している兄弟の数と配偶者の数の和かな

SibSp_surv = pd.crosstab(train['SibSp'],train['Survived'],normalize='index')

print(SibSp_surv)
# SibSp グラフ

height1=SibSp_surv.iloc[:,1]

height2=SibSp_surv.iloc[:,0]

bar1=pyplot.bar(SibSp_surv.index,height1)

bar2=pyplot.bar(SibSp_surv.index,height2,bottom=height1,color="red")

pyplot.title("SibSp-Survive Pct")

pyplot.legend((bar1[0], bar2[0]), ("survived", "dead"))

pyplot.xticks(np.arange(len(SibSp_surv)+2))

pyplot.xlabel("SibSp")
# Parch 同乗している親の数と子の数の和かな

Parch_surv = pd.crosstab(train['Parch'],train['Survived'],normalize='index')

print(Parch_surv)
# Parch グラフ

height1=Parch_surv.iloc[:,1]

height2=Parch_surv.iloc[:,0]

bar1=pyplot.bar(Parch_surv.index,height1)

bar2=pyplot.bar(Parch_surv.index,height2,bottom=height1,color="red")

pyplot.title("Parch-Survive Pct")

pyplot.legend((bar1[0], bar2[0]), ("survived", "dead"))

pyplot.xticks(np.arange(len(Parch_surv)))

pyplot.xlabel("Parch")
# Embarked 出港地、C = Cherbourg, Q = Queenstown, S = Southampton

Embarked_surv = pd.crosstab(train['Embarked'],train['Survived'],normalize='index')

print(Embarked_surv)
# Embarked グラフ

height1=Embarked_surv.iloc[:,1]

height2=Embarked_surv.iloc[:,0]

bar1=pyplot.bar(Embarked_surv.index,height1)

bar2=pyplot.bar(Embarked_surv.index,height2,bottom=height1,color="red")

pyplot.title("Embarked-Survive Pct")

pyplot.legend((bar1[0], bar2[0]), ("survived", "dead"))

pyplot.xlabel("Embarked")
# Fareの30区切りの度数

a = np.arange(0,560,30)

np.histogram(train['Fare'],bins=a)
# Fareの30区切りの度数毎の生存率

Fare_surv = list()

for i in range(0,len(a)-1):

    tmp = train[(train['Fare']>=a[i])&(train['Fare']<a[i+1])]

    b = tmp['Survived'][tmp['Survived']==1].sum()/len(tmp)

    Fare_surv.append((str(a[i])+"-"+str(a[i+1]),1-b,b))

Fare_surv = pd.DataFrame(Fare_surv,columns=["range","0","1"])

Fare_surv.index = Fare_surv['range']
# Fare グラフ

height1=Fare_surv.iloc[:,2]

height2=Fare_surv.iloc[:,1]

bar1=pyplot.bar(Fare_surv.index,height1)

bar2=pyplot.bar(Fare_surv.index,height2,bottom=height1,color="red")

pyplot.title("Fare-Survive Pct")

pyplot.legend((bar1[0], bar2[0]), ("survived", "dead"))

pyplot.xticks(Fare_surv.index,rotation=90)

pyplot.xlabel("Fare")
# Ageの度数

a = np.arange(0,100,10)

np.histogram(train['Age'],bins=a)
# Ageの10区切りの度数毎の生存率

Age_surv = list()

for i in range(0,len(a)-1):

    tmp = train[(train['Age']>=a[i])&(train['Age']<a[i+1])]

    b = tmp['Survived'][tmp['Survived']==1].sum()/len(tmp)

    Age_surv.append((str(a[i])+"-"+str(a[i+1]),1-b,b))

Age_surv=pd.DataFrame(Age_surv,columns=["range","0","1"])

Age_surv.index = Age_surv['range']
# Age グラフ

height1=Age_surv.iloc[:,2]

height2=Age_surv.iloc[:,1]

bar1=pyplot.bar(Age_surv.index,height1)

bar2=pyplot.bar(Age_surv.index,height2,bottom=height1,color="red")

pyplot.title("Age-Survive Pct")

pyplot.legend((bar1[0], bar2[0]), ("survived", "dead"))

pyplot.xticks(Age_surv.index,rotation=90)

pyplot.xlabel("Age")
train['Cabin'][train['Cabin'].notnull()]
# 家族のケース

train[train['Cabin']=="B20"]
# 一番金持ち

train[train['Cabin']=="B51 B53 B55"]
train['Ticket']
train2['Ticket'][1].rsplit(" ",1)[len(train2['Ticket'][1].rsplit(" ",1))-1]





#for i in range(0,len(train2)):

#    if len(train2['Ticket'][i].rsplit(" ",1))==2:

#        train2['Ticket1'][i]=train2['Ticket'][i].rsplit(" ",1)[0]

#        train2['Ticket2'][i]=train2['Ticket'][i].rsplit(" ",1)[1]

#    else:

#        train2['Ticket2'][i]=train2['Ticket'][i].rsplit(" ",1)[0]
# Ticketから数字のみ取り出す(カウントすると同乗者数と同意になりそう)

TicketNo = list(map(lambda x: train2['Ticket'][x].rsplit(" ",1)[len(train2['Ticket'][x].rsplit(" ",1))-1],range(len(train2))))

len(TicketNo)
import seaborn as sns
sns.pairplot(train.iloc[:,2:])
sns.pairplot(test.iloc[:,2:])
train2 = train

train2 = train2.drop('Name',axis=1)

train2 = train2.drop('PassengerId',axis=1)

train2['Sex'] = train2['Sex'].replace("male","0")

train2['Sex'] = train2['Sex'].replace("female","1")

train2['Embarked'] = train2['Embarked'].replace("C","0")

train2['Embarked'] = train2['Embarked'].replace("Q","1")

train2['Embarked'] = train2['Embarked'].replace("S","2")

train2.info()
sns.pairplot(train2,hue='Survived')
train2['Sex']=train2['Sex'].astype(int)
sns.heatmap(train2.corr())