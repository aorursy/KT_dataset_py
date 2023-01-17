# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

train['Sex'].replace(['female','male'],[0,1],inplace=True)
pcmod= XGBRegressor(n_estimators=1000000, learning_rate=0.05)
ytrain= train["Survived"]
test=pd.read_csv("../input/test.csv")

test['Sex'].replace(['female','male'],[0,1],inplace=True)
xtraino= train[["Pclass","Sex", 'Age','SibSp','Parch', 'Fare']]
xtesto=test[["Pclass","Sex", 'Age','SibSp','Parch', 'Fare']]
ytrain=train["Survived"]
traino_X, vali_X, traino_y, vali_y = train_test_split(xtraino, ytrain, random_state = 0)
pcmod.fit(traino_X, traino_y, early_stopping_rounds=5, 

             eval_set=[(vali_X, vali_y)], verbose=False) 
finalp= pcmod.predict(xtesto)
prediso=[]

for numb in finalp:

    if numb >= .5:

        prediso.append(1)

    else:

        prediso.append(0)
output = pd.DataFrame({'PassengerId': test.PassengerId,

                       'Survived': prediso})

output.to_csv('submission1.csv', index=False)