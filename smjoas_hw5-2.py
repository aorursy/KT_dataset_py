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

test_data.head()
gender_data=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

gender_data.head()
train_data['Sex_clean']=train_data['Sex'].astype('category').cat.codes

test_data['Sex_clean']=train_data['Sex'].astype('category').cat.codes
train_data.info()
train_data['Family']=1+train_data['SibSp']+train_data['Parch']

test_data['Family']=1+test_data['SibSp']+test_data['Parch']
train_data['solo']=(train_data['Family']==1)

test_data['solo']=(test_data['Family']==1)
features=['Pclass', 'SibSp', 'Sex_clean', 'Parch', 'Family', 'solo']

label=['Survived']
from sklearn.ensemble import RandomForestClassifier



x_train=train_data[features]

x_test=  test_data[features]

y_train= train_data[label]



clf= RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

clf.fit(x_train,y_train)

gender_data['Survived']= clf.predict(x_test)

gender_data.to_csv('titanic_submission.csv', index=False)

print("Your submission was successfully saved!")