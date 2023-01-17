import csv as csv

import numpy as np

import pandas as pd
# For .read_csv, always use header=0 when you know row 0 is the header row

df = pd.read_csv('../input/train.csv', header=0)


    


    
for i in range(1,4):

    print(i , len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ]))
df['Gender']=4
df.head()
df['Gender']=df['Sex'].map( {'female' : 0, 'male' : 1} ).astype(int)
df.head(3)
df['Port']=df['Embarked'].map( {'S':1, 'Q':2, 'C':3} )
df.head(3)
#Age histogram seems positively skewed, so use median to fill in missing values rather than the mean
#Create an array for gender by Pclass

median_ages=np.zeros((2,3))

median_ages
for i in range(0, 2):

    for j in range(0, 3):

        median_ages[i,j] = df[(df['Gender'] == i) & \

                              (df['Pclass'] == j+1)]['Age'].dropna().median()

        

        



        
median_ages
df['AgeFill']=df['Age']

df.head(3)
for i in range(0, 2):

    for j in range(0, 3):

        df.loc[  (df.Age.isnull() )  & (df.Gender == i) & (df.Pclass == j+1),\

                      'AgeFill'] = median_ages[i,j]
df['AgeisNull'] = pd.isnull(df.Age).astype(int)
df=df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df[df.Age.isnull()]=df.Age.median()
df=df.drop(['PassengerId'], axis=1)
df=df.drop(['AgeFill', 'AgeisNull'], axis=1)
df2=pd.read_csv('../input/test.csv', header=0)

df2['Gender']=df2['Sex'].map({'female':0, 'male':1})
df2['Port']=df2['Embarked'].map({'S':1, 'C':2, 'Q':3})
traindf=df

testdf=df2

testdf=testdf.drop(['Name', 'Sex', 'Cabin', 'PassengerId', 'Ticket', 'Embarked'], axis=1)
testdf.info()
testdf[testdf.Age.isnull()]=testdf.Age.median()
testdf[testdf.Fare.isnull()]=testdf.Fare.median()
testdf.info()
traindf=df
traindf.info()
traindf['Port'][traindf.Port.isnull()]=1
traindf.info()
testdf.info()
# Import the random forest package

from sklearn.ensemble import RandomForestClassifier 



# Create the random forest object which will include all the parameters

# for the fit

forest = RandomForestClassifier(n_estimators = 100)
testdf.info()

traindf.info()
train_data=traindf.values

test_data=testdf.values
# Fit the training data to the Survived labels and create the decision trees

forest = forest.fit(train_data[0::,1::],train_data[0::,0])
# Take the same decision trees and run it on the test data

output = forest.predict(test_data)
submission = pd.DataFrame({

        "PassengerId": df2["PassengerId"],

        "Survived": output

    })

submission.to_csv('titanic.csv', index=False)
output[0:10]