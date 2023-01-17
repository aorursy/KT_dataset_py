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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
total_women = train_data.loc[train_data['Sex']=='female'].shape[0]

print('No of women in total:',total_women)



total_men = train_data.loc[train_data['Sex']=='male'].shape[0]

print('No of men in total:',total_men)



Women = train_data.loc[train_data['Sex']=="female",'Survived']

rate_women = sum(Women)/len(Women)

print('% of women survived:', rate_women*100)



Men = train_data.loc[train_data['Sex']=='male','Survived']

rate_men = sum(Men)/len(Men)

print('% of men survived:', rate_men*100)

from sklearn.ensemble import RandomForestClassifier

Y = train_data["Survived"]

features = ["Sex","Pclass","SibSp","Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators= 100,max_depth=5,random_state=1)

model.fit(X,Y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('mySubmission.csv', index=False)

print("your submission has been successfully output and saved")