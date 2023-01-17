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
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()

#print(train_data.shape)
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

#print(train_data.shape,test_data.shape)

test_data.head()
women=train_data.loc[train_data.Sex=='female']['Survived']

sum(women)/len(women)
men=train_data.loc[train_data.Sex=='male']['Survived']

sum(men)/len(men)

from sklearn.ensemble import RandomForestClassifier





y=train_data["Survived"]

features=["Pclass","Sex","SibSp","Parch"]

X=pd.get_dummies(train_data[features])

X_test=pd.get_dummies(test_data[features])

print(X.head())

print(X_test.head())

model=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=10,random_state=1,max_features='sqrt',n_jobs=-1,verbose=1)

model.fit(X,y)
predict=model.predict(X_test)

predict.shape

#X_test.shape
output=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predict})

output
output.to_csv('mysubmission.csv',index=False)

print("Your submission was successfully saved!")