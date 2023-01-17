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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn import preprocessing
train=pd.read_csv("../input/"+'train.csv')
def Preprocess(ntrain,ntest):

    train=pd.read_csv("../input/"+ntrain+'.csv')

    test=pd.read_csv("../input/"+ntest+'.csv')

    train_y=train['Survived']

    train_X=train.drop(columns='Survived')

    concated=pd.concat([train_X,test],keys=['train','test'],sort=False)

    concated=concated.drop(columns=['Name','Ticket','Cabin'])

    concated['Age']=concated['Age'].fillna(concated['Age'].mean())

    concated['Fare']=concated['Fare'].fillna(concated['Fare'].mean())

    concated['Embarked']=concated['Embarked'].fillna('S')

    concated['Sex'].replace(['male','female'],[1,0],inplace=True)

    concated['Embarked'].replace(['S','Q','C'],[0,1,2],inplace=True)

    concated['Fno']=concated['Parch']+concated['SibSp']

    concated['Age']=preprocessing.scale(concated['Age'])

    concated['Fare']=preprocessing.scale(concated['Fare'])

    concated['Parch']=preprocessing.scale(concated['Parch'])

    concated['SibSp']=preprocessing.scale(concated['SibSp'])

    concated['Fno']=preprocessing.scale(concated['Fno'])

    train_x=concated.loc['train'].values[:,:]

    test_x=concated.loc['test'].values[:,:]

    

    return train_x,train_y,test_x

train_x,train_y,test_x=Preprocess('train','test')

train_x

print(len(train_x))
train_y[701:]
eval_x=train_x[701:]

eval_y=train_y[701:]

len(eval_x)

train_x=train_x[:701]

train_y=train_y[:701]

len(train_x)

print(eval_y)
from sklearn.linear_model import LogisticRegression
LRmodel=LogisticRegression(random_state=0,solver='lbfgs',multi_class='multinomial')

LRprediction=LRmodel.fit(train_x,train_y)

print(LRmodel.score(train_x,train_y))

print(LRmodel.score(eval_x,eval_y))
from sklearn.tree import DecisionTreeClassifier
result=DTmodel.predict(test_x)

output=pd.read_csv('../input/gender_submission.csv')

output['Survived']=result

output.to_csv('output.csv',index=False)
from sklearn.ensemble import RandomForestClassifier
RFmodel=RandomForestClassifier(max_depth=20, random_state=0,n_estimators=100)

RFprediction=RFmodel.fit(train_x,train_y)

RFmodel.score(train_x,train_y)

print(RFmodel.score(train_x,train_y))

print(RFmodel.score(eval_x,eval_y))

#seems better
result=RFmodel.predict(test_x)

output=pd.read_csv('../input/gender_submission.csv')

output['Survived']=result

output.to_csv('output.csv',index=False)