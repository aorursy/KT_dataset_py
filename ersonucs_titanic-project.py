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
women = train_data.loc[train_data.Sex=='female']["Survived"]
women
rate_women = sum(women)/len(women)

print('% of women survived',rate_women*100,'%')
man = train_data.loc[train_data.Sex=='male']["Survived"]
rate_man = sum(man)/len(man)

print('% of  man survived',rate_man*100,'%')
#loding the Randomforestmodel

from sklearn.ensemble import RandomForestClassifier



#defining the Y variable which we want to predict

y = train_data["Survived"]



#defining the Features to find the pattern

features = ["Pclass","Sex","SibSp","Parch"]





#defining the X variable from Training dataset

X = pd.get_dummies(train_data[features])



#testing the X variable from test dataset

X_test = pd.get_dummies(test_data[features])



#defining the model

model = RandomForestClassifier(n_estimators = 100,max_depth = 5,random_state=1)



#fitting the model with X and y Variable

model.fit(X,y)



#prediction based for test dataset

predictions = model.predict(X_test)



#showing the output/ Submitting



output = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived': predictions})

output.to_csv('my_submission.csv',index = False)

print("your submission submitted sucessfully")