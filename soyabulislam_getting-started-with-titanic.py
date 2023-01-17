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
train_data=pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data= pd.read_csv("../input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

percentage= sum(women)/ len(women)

percentage

men= train_data.loc[train_data.Sex == 'male']["Survived"]

percentage_men= sum(men)/ len(men)

percentage_men
from sklearn.ensemble import RandomForestClassifier



features= ["Pclass", "Sex",  "SibSp", "Parch", "Embarked"]

y= train_data["Survived"]

x=pd.get_dummies(train_data[features])

x_test= pd.get_dummies(test_data[features])



model= RandomForestClassifier(random_state=1, n_estimators=200, max_depth=8, criterion='entropy', verbose=1)

model.fit(x,y)

prediction= model.predict(x_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print(output)
print("Your submission was successfully saved!")