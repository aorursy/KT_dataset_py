#This is an entry to the kaggle titanic challenge

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")



train = pd.read_csv("../input/titanic/train.csv")

#Training the model on 'train' dataset

y = train.Survived



#The features the model will train on. I need to see if SibSp is needed

features = ['PassengerID', 'Pclass', 'Sex', 'Age', 'SibSp']

print (train)