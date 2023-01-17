# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



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
train_data.info()
test_data.info()
#Gender Data



women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
#Seating Class Data



first_class = train_data.loc[train_data.Pclass == 1]["Survived"]

rate_1st_class = sum(first_class)/len(first_class)



second_class = train_data.loc[train_data.Pclass == 2]["Survived"]

rate_2nd_class = sum(second_class)/len(second_class)



third_class = train_data.loc[train_data.Pclass == 3]["Survived"]

rate_3rd_class = sum(third_class)/len(third_class)



print("{}% of people survived when in first class".format(rate_1st_class*100))

print("{}% of people survived when in second class".format(rate_2nd_class*100))

print("{}% of people survived when in third class".format(rate_3rd_class*100))
#Embarked Data



cherbourg = train_data.loc[train_data.Embarked == "C"]["Survived"]

rate_cherbourg = sum(cherbourg)/len(cherbourg)



queenstown = train_data.loc[train_data.Embarked == "Q"]["Survived"]

rate_queenstown = sum(queenstown)/len(queenstown)



southampton = train_data.loc[train_data.Embarked == "S"]["Survived"]

rate_southampton = sum(southampton)/len(southampton)



print("{}% survived that boarded from Cherbourg".format(rate_cherbourg*100))

print("{}% survived that boarded from Queenstown".format(rate_queenstown*100))

print("{}% survived that boarded from southampton".format(rate_southampton*100))

from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch","Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")