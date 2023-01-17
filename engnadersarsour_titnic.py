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
path="../input/titanic/train.csv"
data=pd.read_csv(path,index_col='PassengerId')
data
data.columns
data.isna().sum()
data.drop(columns=['Name','Ticket','Cabin'],inplace=True)
age_mean=data.Age.mean()
data.Age.fillna(age_mean,inplace=True)
data.Embarked.value_counts()
data.Embarked.fillna('S',inplace=True)
data.isna().sum()
data.Sex.replace('male',0,inplace=True)

data.Sex.replace('female',1,inplace=True)



data.Embarked.replace('C',0,inplace=True)

data.Embarked.replace('S',1,inplace=True)

data.Embarked.replace('Q',2,inplace=True)





    
from sklearn.model_selection import train_test_split
X=data.drop(columns='Survived')
y=data.Survived

X_train,X_test,y_train,y_test=train_test_split(X,y)
para=list(range(2, 15, 1))

para
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,classification_report,f1_score
for i in para:

    results={}

    dt=DecisionTreeClassifier(max_leaf_nodes=2 ,random_state=1)

    dt.fit(X_train,y_train)

    preds=dt.predict(X_test)

    acc=accuracy_score(y_true=y_test,y_pred=preds)

    f1=f1_score(y_true=y_test,y_pred=preds)

    print(i)

    print(classification_report(y_true=y_test,y_pred=preds))

    results[i]=f1

best_para=max(results,key=results.get)
final_model = DecisionTreeClassifier(max_leaf_nodes=best_para)

final_model.fit(X, y)
test_df = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

test_df
test_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

test_df.Age.fillna(age_mean,inplace=True)

test_df.Fare.fillna(test_df.Fare.mean(),inplace=True)
test_df.Sex.replace('male',0,inplace=True)

test_df.Sex.replace('female',1,inplace=True)



test_df.Embarked.replace('C',0,inplace=True)

test_df.Embarked.replace('S',1,inplace=True)

test_df.Embarked.replace('Q',2,inplace=True)
preds = final_model.predict(test_df)

test_out = pd.DataFrame({

    'PassengerId': test_df.index,

    'Survived': preds

})

test_out.to_csv('submission.csv', index=False)