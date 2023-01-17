import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style

%matplotlib inline
style.use('fivethirtyeight')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.head()
train.isnull().sum()
test.isnull().sum()
train['Name_Length'] = train['Name'].apply(len)
test['Name_Length'] = test['Name'].apply(len)

full_data = [train,test]

for data in full_data:
    #Embarked为null的默认为S
    data['Embarked'] = data['Embarked'].fillna('S')
    #Fare默认为平均值
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    #计算家人数量
    data['FimalySize'] = data['SibSp']+data['Parch']+1
    data['isAlone'] = 0
    data.loc[data['FimalySize']==1,'isAlone'] = 1

    #随机添加默认年龄
    age_avg = data['Age'].mean()
    age_std = data['Age'].std()
    age_null_count = data['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    data['Age'].loc[data['Age'].isnull()] = age_null_random_list
    data['Age'] = data['Age'].astype(int)
a = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']
train.drop(a,axis=1,inplace=True)
test.drop(a,axis=1,inplace=True)
train['Count'] = 1
#对Age，Fare，Name_Length三个字段进行分类
#for a , r in {'Age':10,'Fare':100,'Name_Length':20}.items():
#    bins = [x*r for x in np.arange(-1,int(train[a].max()/r)+2)]
#    train[a+'_cut'] = pd.cut(train[a],bins)
train.loc[(train['Age']<=16),'Age_cut'] = '<=16'
train.loc[((train['Age']<=30)&(train['Age']>16)),'Age_cut'] = '<=30'
#train.loc[((train['Age']<=30)&(train['Age']>20)),'Age_cut'] = '<=30'
train.loc[((train['Age']<=50)&(train['Age']>30)),'Age_cut'] = '<=50'
train.loc[(train['Age']>50),'Age_cut'] = '>50'

train.loc[(train['Name_Length']<=20),'Name_Length_cut'] = '<=20'
train.loc[((train['Name_Length']<=25)&(train['Name_Length']>20)),'Name_Length_cut'] = '<=25'
train.loc[((train['Name_Length']<=30)&(train['Name_Length']>25)),'Name_Length_cut'] = '<=30'
train.loc[(train['Name_Length']>30),'Name_Length_cut'] = '>30'

train.loc[(train['Fare']<=10),'Fare_cut'] = '<=10'
train.loc[((train['Fare']<=15)&(train['Fare']>10)),'Fare_cut'] = '<=15'
train.loc[((train['Fare']<=30)&(train['Fare']>15)),'Fare_cut'] = '<=30'
train.loc[(train['Fare']>30),'Fare_cut'] = '>30'

train.head()
def Survived_Rate(x):
    train_new = train.set_index([x],drop=False)
    train_new.sort_index(inplace=True)
    train_new = train_new.groupby(level=[0])
    survived_rate = train_new.agg({'Count':np.sum,'Survived':np.sum}).Survived/train_new.agg({'Count':np.sum,'Survived':np.sum}).Count
    survived_rate = survived_rate.fillna(0.0)
    return survived_rate

def plt_rate(survived_rate,x):
    plt.figure(figsize=(10,5))
    plt.bar(survived_rate.index,survived_rate.values,width=0.3)
    plt.ylim(0,1)
    plt.xlabel(x)
    plt.ylabel('Survived Rate')
    plt.show()
    
    return None
a = train.columns.drop(['Survived','Age','Fare','Name_Length','Count'])
for i in a:
    plt_rate(Survived_Rate(i),i)
#one-hot-encode
train.drop(['Count','Age_cut','Name_Length_cut','Fare_cut'],inplace=True,axis=1)
train = pd.get_dummies(data = train,columns=['Sex','Embarked'])
test = pd.get_dummies(data = test,columns=['Sex','Embarked'])
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(train.drop(['Survived'],axis=1),train['Survived'])
y_pred_bnb = bnb.predict(test)
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(train.drop(['Survived'],axis=1),train['Survived'])
y_pred_lg = lg.predict(test)

