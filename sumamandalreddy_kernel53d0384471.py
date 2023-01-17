# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import precision_score, recall_score

titanic_input_file_path="../input/titanic/train.csv"



data= pd.read_csv(titanic_input_file_path)

#target column I am going to predict

y= data.Survived

#features on which my prediction depends

cols= data.columns.values

print(cols)

features=["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare"]

X= data[cols].drop(['Survived','Name','Ticket','Cabin','Embarked'],axis=1)

X.fillna(0,inplace=True)

X['Sex'].replace("female",0, inplace=True)

X['Sex'].replace("male",1, inplace=True)

X['SibSp'].astype(float)

age_range = [

    (X['Age']<=15),

     (X['Age'] > 15) & (X['Age'] <= 40)  ,

     (X['Age']>40) ]

choices = [1, 2, 3]

X['Age']=np.select(age_range, choices, default='g1')



























#split into train and test data

train_X,val_X, train_y,val_y= train_test_split(X,y, random_state=1)



# build a model

titanic_model= RandomForestClassifier(n_estimators=100)

#rfc = RandomForestClassifier(n_estimators=600, max_depth=300, max_features='sqrt')

#fit the model

titanic_model.fit(train_X, train_y)



#predict

pred=titanic_model.predict(val_X)

print(pred)

print(titanic_model.score(train_X, train_y))

#mean absolute error



#mae=mean_absolute_error(pred, val_y)
import numpy as np

titanic_test_file_path="../input/titanic/test.csv"

data_full= pd.read_csv(titanic_test_file_path)

data_full['Sex'].replace("female",0, inplace=True)

data_full['Sex'].replace("male",1, inplace=True)

data_full.fillna(0,inplace=True)

pred_full=titanic_model.predict(data_full[features])



#pred_full.loc[pred_full.Survived >0.5,'Survived']=1

#pred_full.loc[pred_full.Survived <0.5,'Survived']=0

#pred_full=np.where(pred_full >0.5, 1,0)

print(pred_full)









output = pd.DataFrame({'PassengerId': data_full.PassengerId,'Survived':pred_full})

output.to_csv('submission.csv', index=False)
