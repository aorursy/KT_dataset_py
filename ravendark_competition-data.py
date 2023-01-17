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
data = pd.read_csv('../input/train.csv',low_memory = False)
data.info()

test.info()
data['Embarked'].value_counts()
data.Cabin.value_counts().count()
'cook'
test = pd.read_csv('../input/test.csv',low_memory = False)
test.head()
test.info()
test.drop(['PassengerId','Cabin'],inplace = True,axis  =1)
data.drop(['PassengerId','Cabin'],inplace = True,axis = 1)
data.info()
data['Embarked'].value_counts()
data['Embarked'].fillna('S',inplace = True)
mean = data['Age'].mean()
mean
std = data['Age'].std()
std

null_Count  = data['Age'].isnull().sum()
null_Count
random_age_float = np.random.uniform(mean-std,mean+std,size = (null_Count,))
random_age_float.max()
random_age.max()
data['Age'][pd.isnull(data['Age'])] = random_age_float
data.info()
data['Age'].value_counts().count()
mean = test['Age'].mean()

std = test['Age'].std()

null = test['Age'].isnull().sum()
radn = np.random.uniform(mean-std,mean+std,size = null)
test['Age'][pd.isnull(test['Age'])] = radn
test.info()
data['Name_length'] = data['Name'].apply(lambda x: len(x))
'goole'
data.Name.value_counts().count()
data.drop('Name',axis = 1,inplace = True)

test.drop('Name',axis = 1,inplace = True)
data.info()
data['Sex'] = data['Sex'].map({"female":0,"male":1})
test['Sex'] = test['Sex'].map({"male":1,"female":0})
data['Embarked'] = data['Embarked'].map({"S":3,"C":2,"Q":1})

test['Embarked'] = test['Embarked'].map({"S":3,"C":2,"Q":1})
data['Embarked'].value_counts()
data['SibSp'].value_counts()
data['Family_size']  = data['SibSp']+data['Parch']

test['Family_size'] = test['SibSp']+ data['Parch']
dro = ['SibSp','Parch']

data.drop(dro,axis  = 1,inplace = True)

test.drop(dro,axis = 1,inplace = True)
data.info()
test.info()
data['fam_isnull'] = data['Family_size'].apply(lambda x: True if x==0 else False )
data['fam_isnull'].value_counts()
data['Family_size'].value_counts()
data.info()
test.info()
data.drop('Survived',axis = 1,inplace =True)
f = pd.read_csv('../input/train.csv')
y = f.Survived
data.Ticket.value_counts().count()
import category_encoders as ce
enc  = ce.BinaryEncoder(cols = 'Ticket')
new_train = enc.fit_transform(data)
new_test = enc.fit_transform(test)
new_test.info()
new_test.Fare.mean()
new_test.Fare = new_test['Fare'].fillna(35.6271)
new_train.info()
from sklearn.ensemble import RandomForestClassifier
new_train.to_csv('train_save.csv',index = False)

new_test.to_csv('test_save.csv',index = False)

data.to_csv('pre_train.csv',index = False)

test.to_csv('pre_test.csv',index = False)
model = RandomForestClassifier(n_estimators = 60,max_features = 0.5,n_jobs = -1,max_depth = 2)
model.fit(new_train,y)
eds.value_counts()
from sklearn.feature_extraction import FeatureHasher
encc = FeatureHasher(n_features = 5,input_type = 'string')
new_test_fh = encc.fit_transform(data['Ticket'])
test.info()
test.Fare = test['Fare'].fillna(36.6232)
yo = pd.DataFrame(new_test_fh.toarray())
trainee  = pd.concat([pd.DataFrame(new_test_fh.toarray()),data],axis  =1)
testii.info()
trainee.info()
trainee.to_csv('final_train.csv',index =False)
testii.to_csv('final_test.csv',index = False)
model.fit(trainee,y)
testii.drop('Ticket',axis  = 1,inplace = True)
y_pred = model.predict(testii)
de = pd.read_csv('../input/test.csv')

StackingSubmission = pd.DataFrame({ 'PassengerId': de.PassengerId,

                            'Survived': y_pred })
StackingSubmission.to_csv("submission.csv", index=False)
'final_data_yo'
testii.to_csv('testii_save.csv',index = False)
!<a href="final_test_modle.csv"> Download File </a>
os.chdir("/kaggle/working/")
!ls /kaggle/working
def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

create_download_link(df)

from IPython.display import FileLink
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64
FileLink('trainee_train_without_index.csv')
trainee.to_csv('trainee_train_without_index.csv',index =False)
testii.to_csv('with_index_testii.csv')