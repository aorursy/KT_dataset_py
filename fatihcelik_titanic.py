import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



#First, import the training and test data.

df = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
df.head()
l = []

for i in df["SibSp"]:

    if(i >=1):

        l.append(0)

    else:

        l.append(1)

l = pd.Series(l)

df.drop(["SibSp"],axis=1,inplace=True)

df["is_Alone"] = l

df.head()
#Same process on the test_data

l = []

for i in test_data["SibSp"]:

    if(i >=1):

        l.append(0)

    else:

        l.append(1)

l = pd.Series(l)

test_data.drop(["SibSp"],axis=1,inplace=True)

test_data["is_Alone"] = l

test_data.head()
def statue(name):

    l = name.split()

    for i in l:

        if(i == "Mrs."):

            return "Mrs."

        elif(i == "Miss."):

            return "Miss."

        elif(i == "Mr."):

            return "Mr."

statues = []  

for i in df["Name"]:

    statues.append(statue(i))

statues = pd.Series(statues)

df["Statues"] = statues



#We have some None values in the "Statues" column. I'll fill it with most_frequent element in this column.

df["Statues"].fillna(df["Statues"].value_counts().index[0],inplace=True)

df.head()
#I'll apply same process on test data

def statue(name):

    l = name.split()

    for i in l:

        if(i == "Mrs."):

            return "Mrs."

        elif(i == "Miss."):

            return "Miss."

        elif(i == "Mr."):

            return "Mr."

statues = []  

for i in test_data["Name"]:

    statues.append(statue(i))

statues = pd.Series(statues)

test_data["Statues"] = statues



#We have some None values in the "Statues" column. I'll fill it with most_frequent element in this column.

test_data["Statues"].fillna(test_data["Statues"].value_counts().index[0],inplace=True)

test_data.head()
id = 0

new_class = []

while id < 891:

    if(df["Pclass"][id] == 1):

        new_class.append("High")

    elif(df["Pclass"][id] == 2):

        new_class.append("Medium")

    else:

        new_class.append("Low")

    id += 1

new_class = pd.Series(new_class)

df["New_Class"] = new_class

df.drop(["Pclass"],axis=1,inplace=True)

df.head()
#Same process on the test_data.

id = 0

new_class = []

while id < 418:

    if(test_data["Pclass"][id] == 1):

        new_class.append("High")

    elif(test_data["Pclass"][id] == 2):

        new_class.append("Medium")

    else:

        new_class.append("Low")

    id += 1

new_class = pd.Series(new_class)

test_data["New_Class"] = new_class

test_data.drop(["Pclass"],axis=1,inplace=True)

test_data.head()
#Unsuccessfull : )



# #Let's look at the "Age" feature closer.

# #Maybe, we can create 4 category based on "Age"

# #0  - 16 

# #16 - 32

# #32 - 48

# #48 - 64

# #64 - 80

# #print (df[['Age', 'Survived']].groupby(['Age']).mean())



# New_Age = []

# id = 0

# while id < 891:

#     if(df["Age"][id] < 16):

#         New_Age.append(1)

#     elif(df["Age"][id] < 32):

#         New_Age.append(2)

#     elif(df["Age"][id] < 48):

#         New_Age.append(3)

#     elif(df["Age"][id] < 64):

#         New_Age.append(4)

#     elif(df["Age"][id] <= 80):

#         New_Age.append(5)

#     id += 1



# New_Age = pd.Series(New_Age)

# df["New_Age"] = New_Age

# df.drop(["Age"],axis=1,inplace=True)

# df["New_Age"].fillna(df["New_Age"].value_counts().index[0],inplace=True)

# df.head()
#Unsuccessfull : )



# #Same process on test_data



# New_Age = []

# id = 0

# while id < 418:

#     if(test_data["Age"][id] < 16):

#         New_Age.append(1)

#     elif(test_data["Age"][id] < 32):

#         New_Age.append(2)

#     elif(test_data["Age"][id] < 48):

#         New_Age.append(3)

#     elif(test_data["Age"][id] < 64):

#         New_Age.append(4)

#     elif(test_data["Age"][id] <= 80):

#         New_Age.append(5)

#     id += 1



# New_Age = pd.Series(New_Age)

# test_data["New_Age"] = New_Age

# test_data.drop(["Age"],axis=1,inplace=True)

# test_data["New_Age"].fillna(test_data["New_Age"].value_counts().index[0],inplace=True)

# test_data.head()
#Unsuccessful : )

#I'll not delete these blocks intetionally.So I can notice my mistakes.If you don't want to read blocks like that,

#You can skip,if block has "Unsuccessful" header.



# #Let's look at the "Fare" feature. I'll create new column based on "Fare" that will have four category.

# #Min -> %25 --> 0

# #%25 -> %50 --> 1

# #%50 -> %75 --> 2

# #%75 -> Max --> 3

# df["Fare"].describe()
#Unsuccessful : )



# New_Fare = []

# id = 0

# while id < 891:

#     if(df["Fare"][id] < 8):

#         New_Fare.append(0)

#     elif(df["Fare"][id] < 14.5):

#         New_Fare.append(1)

#     elif(df["Fare"][id] < 31):

#         New_Fare.append(2)

#     elif(df["Fare"][id] > 31):

#         New_Fare.append(3)

#     id += 1

# New_Fare = pd.Series(New_Fare)

# df["New_Fare"] = New_Fare

# df.drop(["Fare"],axis=1,inplace=True)



# #Impute with most_frequent Nan datas.

# df["New_Fare"].fillna(df["New_Fare"].value_counts().index[0],inplace=True)

# df.head()
#Unsuccessful : )



# New_Fare = []

# id = 0

# while id < 418:

#     if(test_data["Fare"][id] < 8):

#         New_Fare.append(0)

#     elif(test_data["Fare"][id] < 14.5):

#         New_Fare.append(1)

#     elif(test_data["Fare"][id] < 31):

#         New_Fare.append(2)

#     elif(test_data["Fare"][id] > 31):

#         New_Fare.append(3)

#     id += 1

# New_Fare = pd.Series(New_Fare)

# test_data["New_Fare"] = New_Fare

# test_data.drop(["Fare"],axis=1,inplace=True)

# #Impute with most_frequent Nan datas.

# test_data["New_Fare"].fillna(test_data["New_Fare"].value_counts().index[0],inplace=True)

# test_data.head()
y = df["Survived"]

df.drop(["Survived","Name","Ticket"],axis=1,inplace=True)

test_data.drop(["Name","Ticket"],axis=1,inplace=True)
df.info()
df.head()
df.describe()
nan_values = ["Column Name: {} -> None Values: {} -> Type: {}".format(column,df[column].isnull().sum(),df[column].dtype) for column in df.columns]

nan_values
df.drop(["Cabin"],axis=1,inplace=True)

test_data.drop(["Cabin"],axis=1,inplace=True)

df.columns
df["Age"].fillna((df["Age"].mean()),inplace=True)

df["Age"].isnull().sum() #None values -> 0. We impute them with its mean.



test_data["Age"].fillna((df["Age"].mean()),inplace=True)

test_data["Age"].isnull().sum() #None values -> 0. We impute them with its mean.
df["Embarked"].fillna(df["Embarked"].value_counts().index[0],inplace=True)

df["Embarked"].isnull().sum()



test_data["Embarked"].fillna(test_data["Embarked"].value_counts().index[0],inplace=True)

test_data["Fare"].fillna(test_data["Fare"].value_counts().index[0],inplace=True)

test_data["Embarked"].isnull().sum()

df1 = pd.get_dummies(df)

test_data = pd.get_dummies(test_data)

df1.head()
df1.drop(["PassengerId"],axis=1,inplace=True)

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(df1,y,test_size=0.3,random_state=2)
from sklearn.metrics import accuracy_score

import xgboost as xgb

xgboost = xgb.XGBClassifier(max_depth=15, n_estimators=400, learning_rate=0.02).fit(train_x, train_y)

xgb_prediction = xgboost.predict(test_x)

xgb_score=accuracy_score(test_y, xgb_prediction)

print(xgb_score)
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100,random_state=0)

clf.fit(train_x, train_y)



prediction = clf.predict(test_x)

score=accuracy_score(test_y, prediction)

print(score)

    
test = test_data.drop(["PassengerId"],axis=1)

prediction = clf.predict(test)

submission = pd.DataFrame({

       "PassengerId": test_data["PassengerId"],

       "Survived": prediction

   })

submission.to_csv('titanic.csv', index=False)