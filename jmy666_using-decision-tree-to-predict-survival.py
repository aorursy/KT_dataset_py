#import



import numpy as np

import pandas as pd

from pandas import Series,DataFrame

import matplotlib as mpl

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn import tree
#read data



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.head()
test.head()
train.info()

print('----------------------')

test.info()
#drop useless columns

#since cabin has too many nulls, we need to delete it

#Also drop passenger_id, ticket, name, and embarked since logically they are useless in prediction

#Fare is related to class, so we delete it

train_p = train.drop(["Cabin","PassengerId","Ticket","Name","Embarked","Fare"],axis=1)

test_p = test.drop(["Cabin","PassengerId","Ticket","Name","Embarked","Fare"],axis=1)

train_p.head()

train_p.info()
#Age

#Logically, when accidents occur, we normally give priority to children and old people. 

#Therefore, I divide age into 3 groups:(,16),(16,60),(60,) to represent child, adult and old

#since most passengers are adults, we fill nulls with adult

#0:child;1:adult,2:old

train_p['Age'].dropna().hist(bins=70)

plt.show()



train_p['Age_new'] = 1

train_p['Age_new'][train_p["Age"]<16] = 0

train_p['Age_new'][train_p["Age"]>60] = 2

train_p['Age_new']=train_p['Age_new'].astype(int)

train_p['Age_new'].hist()

plt.show()        

train_p.info()



test_p['Age_new'] = 1

test_p['Age_new'][test_p["Age"]<16] = 0

test_p['Age_new'][test_p["Age"]>60] = 2

test_p['Age_new']=test_p['Age_new'].astype(int)

test_p['Age_new'].hist()

plt.show()
#Family



#Referrence:https://www.kaggle.com/omarelgabry/a-journey-through-titanic

# Instead of having two columns Parch & SibSp, 

# we can have only one column to represent if the passenger had any family member aboard or not,

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

train_p['Family'] =  train_p["Parch"] + train_p["SibSp"]

train_p['Family'].loc[train_p['Family'] > 0] = 1

train_p['Family'].loc[train_p['Family'] == 0] = 0



test_p['Family'] =  test_p["Parch"] + test_p["SibSp"]

test_p['Family'].loc[test_p['Family'] > 0] = 1

test_p['Family'].loc[test_p['Family'] == 0] = 0





# drop Parch & SibSp

train_p = train_p.drop(['SibSp','Parch'], axis=1)

test_p    = test_p.drop(['SibSp','Parch'], axis=1)



#gender

#male is 1

train_p['Gender'] = 0

train_p['Gender'][train_p['Sex']=="male"] = 1



test_p['Gender'] = 0

test_p['Gender'][test_p['Sex']=="male"] = 1

#final data preparation

train_y = train_p['Survived']

train_x = train_p.drop(['Survived','Age',"Sex"],axis=1)



test_x = test_p.drop(['Age','Sex'],axis=1)



train_y.head()

train_x.head()

test_x.head()

train_x.info()

#decision tree model

model_tree = tree.DecisionTreeClassifier()

model_tree = model_tree.fit(train_x, train_y)



print(model_tree.feature_importances_)

print(model_tree.score(train_x, train_y))

#The score of [Pclass	Age_new	Family	Gender] is:

#[ 0.24949133  0.08455378  0.01735779  0.6485971 ]

#The final score is 0.81593714927
#predict

a = model_tree.predict(test_x)



test_x["Survived"] = a



test_data = test_x

test_data.head()
submission_data = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':test_x['Survived']})



submission_data.to_csv("submission_data.csv",index=0)