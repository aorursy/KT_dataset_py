# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



#Print you can execute arbitrary python code

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

df = train

print (train.head())
print (train.groupby('Pclass').mean())
class_sex_grouping = (train.groupby(['Pclass','Sex']).mean())

print (class_sex_grouping)
import matplotlib.pyplot as plt



class_sex_grouping['Survived'].plot.bar()

plt.show()
group_by_age = pd.cut(train["Age"], np.arange(0, 90, 10))

age_grouping = train.groupby(group_by_age).mean()

age_grouping['Survived'].plot.bar()

plt.show()
print (train.info())
train = train.drop(['Name', 'Cabin', 'Ticket'], axis=1) 

train = train.dropna()

print (train.head())
print (train.info())
#Create matrix for random forest classifier

train.fillna(0, inplace=True)

test.fillna(0, inplace=True)

train.replace({'female':1,'male':0, 'S':1, 'C':2, 'Q':3}, inplace=True)

test.replace({'female':1,'male':0, 'S':1, 'C':2, 'Q':3}, inplace=True)



cols = ['Pclass','Age','SibSp', 'Embarked','Sex']

x_train = train[cols]

y_train = train['Survived']

#x_train, x_test, y_train, y_test = train_test_split(train[cols], train['Survived'], test_size=0.75, random_state=42)

x_test = test[cols]

id_test = test['PassengerId']



print("Training samples: {}".format(len(x_train)))

print("Testing samples: {}".format(len(y_train)))



#initialize the model

model = RandomForestClassifier(n_estimators=100)

model.fit(x_train, y_train)

score = cross_val_score(model, x_train, y_train)

print("RandomForestClassifier :")

print(score)



output = pd.DataFrame(model.predict(x_test))

print(type(output))

print(type(id_test))

submission = pd.concat([id_test,output],axis=1)

submission.columns = ['PassengerId', 'Survived']



#Any files you save will be available in the output tab below

submission.to_csv('submission.csv', index=False)