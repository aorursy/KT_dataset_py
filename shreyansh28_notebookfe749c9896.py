# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv("../input/train.csv")

print(train_data.shape)

print(train_data.head())



test_data=pd.read_csv("../input/test.csv")

print(test_data.shape)

print(test_data.tail())

PassId=test_data.iloc[:,0:1]



del test_data['PassengerId'],test_data['Name'],test_data['Ticket'],test_data['Cabin']

test_data['Sex'][test_data['Sex']=='female']=0

test_data['Sex'][test_data['Sex']=='male']=1

test_data['Embarked'][test_data['Embarked']=='S']=0

test_data['Embarked'][test_data['Embarked']=='C']=1

test_data['Embarked'][test_data['Embarked']=='Q']=2

test_data['Embarked'].fillna(0,inplace=True)

test_data['Age'].fillna(np.mean(test_data['Age']),inplace=True)

test_data['family']=test_data['Parch']+test_data['SibSp']

del test_data['SibSp'],test_data['Parch']

print(test_data.info())


test_data['Fare'].fillna(np.mean(test_data['Fare']),inplace=True)

print(test_data)
df_a=train_data.iloc[:,4:]

df_b=train_data.iloc[:,2:3]

df_c = pd.concat([df_b, df_a], axis=1)

del df_c['Ticket']

del df_c['Cabin']





train_target=train_data.iloc[:,1:2]

df_c['Sex'][df_c['Sex']=='female']=0

df_c['Sex'][df_c['Sex']=='male']=1

df_c['Embarked'][df_c['Embarked']=='S']=0

df_c['Embarked'][df_c['Embarked']=='C']=1

df_c['Embarked'][df_c['Embarked']=='Q']=2

df_c['Embarked'].fillna(0,inplace=True)

df_c['Age'].fillna(np.mean(df_c['Age']),inplace=True)



print(df_c.head())

df_c.info()



plt.plot(df_c['Age'],'.')

#model=LogisticRegression()

#model.fit(X,y)

df_c['family']=df_c['Parch']+df_c['SibSp']

del df_c['SibSp'],df_c['Parch']

print(df_c.head)


plt.plot(df_c['Age'],'.')

df_c['Age'].fillna(np.mean(df_c['Age']),inplace=True)

df_c.info()
model = LogisticRegression()

model = model.fit(X, y)

print(df_c.head())


print(train_target.values.ravel())
#X_train, X_test, y_train, y_test = train_test_split(df_c,train_target.values.ravel() , test_size=0.2, random_state=0)
model=LogisticRegression()

#model.fit(X_train,y_train)

model.fit(df_c,train_target.values.ravel())
predicted=model.predict(test_data)

print(predicted)

#print(np.mean(predicted==y_test))

df=pd.DataFrame(predicted)

df

df.columns=['Survived']

final=pd.concat([PassId, df], axis=1)

print(final)

final.to_csv('gender_submission.csv', header=True,index=False)



print(check_output(["ls"]).decode("utf8"))
