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
from sklearn.tree import DecisionTreeRegressor
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
y=train_data.Survived

features=['Pclass','Parch','Sex']

X=pd.get_dummies(train_data[features])

X_val=pd.get_dummies(test_data[features])
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)

#tit_model=DecisionTreeRegressor(max_leaf_nodes=15,random_state=1)

tit_model=RandomForestRegressor(max_leaf_nodes=9,random_state=1)

tit_model.fit(train_X,train_y)



predictions=tit_model.predict(val_X)

#print(mean_absolute_error(predictions,val_y))

#predictions=np.around(predictions)



predictions=tit_model.predict(X_val)

predictions=np.around(predictions)

predictions=pd.to_numeric(predictions,downcast='integer')



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#print(output)

output.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")