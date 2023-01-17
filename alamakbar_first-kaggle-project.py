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
#loading data for training

training_data = pd.read_csv("/kaggle/input/titanic/train.csv")

training_data.head()
#loading all testing data

testing_data = pd.read_csv("/kaggle/input/titanic/test.csv")

testing_data.head()
#checking null values in data

testing_data.isnull()
#women survival rate

women = training_data.loc[training_data.Sex == 'female']["Survived"]

women_surv_rate = sum(women)/ len(women)

print("% wise women who survived:", women_surv_rate)
#men survival rate

men = training_data.loc[training_data.Sex == 'male']["Survived"]

men_surv_rate = sum(men) / len(men)

print("% wise men who survived:", men_surv_rate)
#creating our first model using random forest classifier 

from sklearn.ensemble import RandomForestClassifier



y = training_data['Survived']



features = ['Sex', 'SibSp', 'Pclass', 'Parch']



X = pd.get_dummies(training_data[features])

X_test = pd.get_dummies(testing_data[features])



model = RandomForestClassifier(n_estimators = 100 , max_depth = 5, random_state = 1)



model.fit(X, y)



prediction = model.predict(X_test)



output = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': prediction})



output.to_csv("my_first_submission.csv", index = False)

print("First submission was succesfull")