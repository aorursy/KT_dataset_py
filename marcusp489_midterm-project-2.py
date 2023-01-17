

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sbn



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn 



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import SGDClassifier

import os

print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.describe()
trnlen = len(train)

data = pd.concat(objs =[train,test],axis = 0,sort = True).reset_index(drop = True)



data.isnull().sum()

data.head()
data['Embarked'].fillna(data['Embarked'].mode())

data['Fare'].fillna(data['Fare'].median())

data['Cabin'].fillna("None")
agecorr = sbn.heatmap(data.corr(),annot = True)


data['Age'].fillna(np.NaN)

for i in range(0,len(data['Age'])-1):

    if np.isnan(data['Age'].iloc[i]):

        mypclass = data['Pclass'].iloc[i]

        myss = data['SibSp'].iloc[i]

        simdata = data[(data['Pclass']== mypclass)&(data['SibSp']== myss)]['Age']

        if (simdata.size == 0): #if there are no similar tuples

            data['Age'].iloc[i] = data['Age'].median() 

        else:

            data['Age'].iloc[i] = simdata.median()
data["Mr"] = 0

data["Mrs"] = 0

data["Ms"] = 0

for i in range(0,len(data)-1):

    myname = data['Name'].iloc[i]

    if("Don" in myname or "Major" in myname or "Capt" in myname or "Jonkheer" in myname or "Rev" in myname or "Col" in myname):

        data["Mr"].iloc[i] = 1

    elif("Countess" in myname or "Mme" in myname):

        data["Mrs"].iloc[i] = 1

    elif("Mlle" in myname or "Ms" in myname):

        data["Ms"].iloc[i] = 1

    elif("Doctor" in myname):

        if(data['Sex'].iloc[i]  == 'female'):

            data["Mrs"].iloc[i] = 1 #the only female doctor aboard was a married woman

        else:

            data["Mr"].iloc[i] = 1

data.describe()
data['FSize'] = 0

for i in range(0,len(data)-1):

    data['FSize'].iloc[i] = data['SibSp'].iloc[i] + data['Parch'].iloc[i]

data.head()
for col in data.columns:

    if data[col].dtype == 'object' and col != 'Name' and col!= 'Ticket':

        lbl = LabelEncoder()

        lbl.fit(list(data[col].values))

        data[col]= lbl.transform(list(data[col].values))

    if (col != 'Name') and (col != 'PassengerId') and (col != 'Ticket') and (data[col].max() > 1 or data[col].min() < 0):

        sclr = MinMaxScaler()#defaults to 0-1

        data[col] = sclr.fit_transform(data[col].values.reshape(-1,1))

data.describe()

train = data[:trnlen]

test = data[trnlen:]

y = train['Survived']

features =['Age','Cabin','Embarked','Fare','Parch','Pclass','Sex','SibSp','Mr','Ms','Mrs','FSize']

X = train[features]

sgd = SGDClassifier()

sgd.fit(X,y)

submsn = pd.DataFrame()

output = sgd.predict(test[features].fillna(0)).astype(int)

submsn['PassengerId']=test['PassengerId']

submsn['Survived']= output.astype(int)

submsn.shape

submsn.to_csv('submission.csv',index = False)
