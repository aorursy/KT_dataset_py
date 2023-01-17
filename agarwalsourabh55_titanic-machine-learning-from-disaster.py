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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.info()
women = train_data.loc[train_data.Sex == 'female']['Survived']

#print(women)

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
train_data.info()
train_data=train_data.drop(['Name','Ticket'],axis=1)

test_data=test_data.drop(['Name','Ticket'],axis=1)

#age = train_data.loc[train_data.Age== 'nan']["Survived"]

#sum(age)/len(age)

#print(age)





train_data_age_test=train_data[train_data['Age'].isnull()]

a=train_data_age_test.index

a=list(a)

b=[]

for i in range(0,891):

    if i not in a:

        b.append(i)

train_data_age_train=train_data.loc[b,:]

train_data_age_train=train_data_age_train.drop(['Cabin'],axis=1)

train_data_age_test=train_data_age_test.drop(['Cabin'],axis=1)



train_data_age_train['Embarked'].fillna(train_data_age_train['Embarked'].value_counts().index[0],inplace=True)

train_data_age_test['Embarked'].fillna(train_data_age_test['Embarked'].value_counts().index[0],inplace=True)



train_data_age_train=pd.get_dummies(train_data_age_train,drop_first=True)

train_data_age_test=pd.get_dummies(train_data_age_test,drop_first=True)



train_data_age_test=train_data_age_test.drop(['Age'],axis=1)

target_age=train_data_age_train['Age']

train_data_age_train=train_data_age_train.drop(['Age'],axis=1)



from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()

rf.fit(train_data_age_train,target_age)

predit=rf.predict(train_data_age_test)



train_data_age_test['Age']=predit

train_data_age_train['Age']=target_age



train_data=pd.concat([train_data_age_train,train_data_age_test])
#age = train_data.loc[train_data.Age== 'nan']["Survived"]

#sum(age)/len(age)

#print(age)





test_data_age_test=test_data[test_data['Age'].isnull()]

a=test_data_age_test.index

a=list(a)

b=[]

for i in range(0,418):

    if i not in a:

        b.append(i)

test_data_age_train=test_data.loc[b,:]



#delete unneccesary column as it cotain too much of Nan Vlue 

test_data_age_train=test_data_age_train.drop(['Cabin'],axis=1)

test_data_age_test=test_data_age_test.drop(['Cabin'],axis=1)



#fill the embarked column

test_data_age_train['Embarked'].fillna(test_data_age_train['Embarked'].value_counts().index[0],inplace=True)

test_data_age_test['Embarked'].fillna(test_data_age_test['Embarked'].value_counts().index[0],inplace=True)



test_data_age_train['Fare'].fillna(test_data_age_train['Fare'].value_counts().index[0],inplace=True)

test_data_age_test['Fare'].fillna(test_data_age_test['Fare'].value_counts().index[0],inplace=True)



#handle the categorical value 

test_data_age_train=pd.get_dummies(test_data_age_train,drop_first=True)

test_data_age_test=pd.get_dummies(test_data_age_test,drop_first=True)



#separtating taret value and input variable 

test_data_age_test=test_data_age_test.drop(['Age'],axis=1)

target_age=test_data_age_train['Age']

test_data_age_train=test_data_age_train.drop(['Age'],axis=1)



from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()



rf.fit(test_data_age_train,target_age)

predicct=rf.predict(test_data_age_test)



test_data_age_test['Age']=predicct

test_data_age_train['Age']=target_age



test_data=pd.concat([test_data_age_train,test_data_age_test])
#train_data.info()
#train_data_age_test.info()

#train_data_age_train.info()



target=train_data['Survived']

train_data=train_data.drop(['Survived'],axis=1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline













#y = train_data["Survived"]

#features = ["Pclass", "Sex", "SibSp", "Parch","Fare"]

#X= pd.get_dummies(train_data[features])

#from sklearn.preprocessing import MinMaxScaler







#X_train = scaler.transform(X_train)

#X_test = scaler.transform(X_test)



X_train,X_test,y_train,y_test=train_test_split(train_data,target,test_size=0.2)



#scaler = MinMaxScaler(feature_range = (0,40)).fit(X_train)



#X_train = scaler.transform(X_train)

#X_test = scaler.transform(X_test)



#X1_test = pd.get_dummies(test_data[features])



#from sklearn.impute import SimpleImputer

#my_imputer = SimpleImputer()

#X1_test = my_imputer.fit_transform(X1_test)



#scaler1=MinMaxScaler(feature_range=(0,40)).fit(X1_test)

#X1_test=scaler1.transform(X1_test)

param_grid = [

    {'classifier' : [LogisticRegression()],

     'classifier__penalty' : ['l1', 'l2'],

    'classifier__C' : np.logspace(-4, 4, 20),

    'classifier__solver' : ['liblinear']},

    {'classifier' : [RandomForestClassifier()],

    'classifier__n_estimators' : list(range(10,101,10)),

    'classifier__max_features' : list(range(6,32,5))}

]



pipe = Pipeline([('classifier' , RandomForestClassifier())])



model = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

test_data
model.fit(X_train, y_train)



predictions = model.predict(X_test)

predict1=model.predict(X_test)

print(accuracy_score(predict1,y_test))

predictions=model.predict(test_data)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission1.csv', index=False)

print("Your submission was successfully saved!")