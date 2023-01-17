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
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



# We will split the train data into two groups: 

# one used to fit the model and one used to evaluate the accruacy using MAE



train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 0)



#Import MAE to verify the accurracy of the model

from sklearn.metrics import mean_absolute_error



model = RandomForestClassifier(n_estimators=70, max_depth=5, random_state=1)

model.fit(train_X, train_y)

pred_y = model.predict(val_X)

mae = mean_absolute_error(val_y,pred_y)



print(mae)









#Now we fit in all the data and out put the prediction 



model = RandomForestClassifier(n_estimators=70, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")