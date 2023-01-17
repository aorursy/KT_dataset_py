#import the below mathematical libraries
import pandas as pd
import numpy as np
import csv as csv

#import libraries for plotting data
import matplotlib.pyplot as plt
import seaborn as sb

#import machine learning libraries
from sklearn.ensemble import RandomForestClassifier 
#Read in the train.csv file
train_data = pd.read_csv("../input/train.csv")

#Print the columns and their corresponding types
print(train_data.info())

#Print a summary of all the numerical fields
print(train_data.describe())
# We have to temporarily drop the rows with 'NA' values
# because the Seaborn plotting function does not know
# what to do with them

sb.pairplot(train_data[['Sex','Pclass', 'Age', 'Survived']].dropna(), hue='Sex')
#I'll first calculate the median age for each pclass field and insert it into a dictionary
#Create dictionary and list of all the unique pclass values
pclass_values = train_data['Pclass'].unique()
median_age_pclass = {}

#Loop through all pclass values and calculate the median for each of them
for i in pclass_values:
    median_age_pclass[i] = train_data[train_data['Pclass'] == i]['Age'].median()

#print dictionary to check results are as expected
print(median_age_pclass)

#We now have the median value for each task!
#Create the new age_all field which is initially a direct copy of Age
train_data['Age_All'] = train_data['Age']

#Create the Age_Estimated Flag
train_data['Age_Estimated'] = pd.isnull(train_data['Age']).astype(int)

#Now Update all instances where Age_Estimated flag = 1 with the corresponding Pclass median
for i in pclass_values:
    train_data.loc[(train_data['Age_Estimated'] == 1) & (train_data['Pclass'] == i),'Age_All'] = median_age_pclass[i]

#Print the top 20 rows of a few select columns to spot check if they correspond accordingly
print(train_data[['Sex','Pclass','Age','Age_All','Age_Estimated','Survived']].head(20))

#I'll first calculate the median age for each pclass field and insert it into a dictionary
#Create dictionary and list of all the unique pclass values
median_fare_pclass = {}

#Create subset of data where Fare > 0
td_fare = train_data.loc[train_data['Fare'] >0,('Pclass','Fare')]

#Loop through all pclass values and calculate the median for each of them
for i in pclass_values:
    median_fare_pclass[i] = td_fare[td_fare['Pclass'] == i]['Fare'].median()
    
#Create the new age_all field which is initially a direct copy of Age
train_data['Fare_All'] = train_data['Fare']

#Now Update all instances where Age_Estimated flag = 1 with the corresponding Pclass median
for i in pclass_values:
    train_data.loc[((train_data['Fare_All'] == 0) | (train_data['Fare_All'].isnull())) & (train_data['Pclass'] == i),'Fare_All'] = median_fare_pclass[i]    

#Print the all rows where Fare = 0
print(train_data.loc[(train_data['Fare'] == 0),('Sex','Pclass','Fare','Fare_All','Survived')])
#Create a mapping dictionary
Sex_Map = {'female': 1,'male': 2}
Embarked_Map = {'Q':1,'C':2,'S':3,'X':0}

#Set all NaN values in Embarked Field = X
train_data.loc[(train_data['Embarked'].isnull()),'Embarked'] = 'X'

#Use mapping dictionary to map values from 'Sex' field to the new field 'Gender'
train_data['Gender'] = train_data['Sex'].map(Sex_Map).astype(int)

#Use mapping dictionary to map values from 'Embarked' field to the new field 'Embarked_From'
train_data['Embarked_From'] = train_data['Embarked'].map(Embarked_Map).astype(int)

#Print the top 20 rows of a few select columns to spot check if they correspond accordingly
print(train_data[['Sex','Gender','Embarked','Embarked_From','Survived']].head(20))
#Print the columns and their corresponding types
print(train_data.info())
#Make the clean dataset and put it into a NumPy array
train = train_data[['Survived','Pclass','Age_All','Age_Estimated','Gender','SibSp','Parch','Fare_All','Embarked_From']].values

#Print top 5 rows of NumPy array to double check the format is correct
print(train[:][0:5])
#Read in the test.csv file
test_data = pd.read_csv("../input/test.csv")

#I'll first calculate the median age for each pclass field and insert it into a dictionary
#Create dictionary and list of all the unique pclass values
median_test_pclass = {}

#Loop through all pclass values and calculate the median for each of them
for i in pclass_values:
    median_test_pclass[i] = test_data[test_data['Pclass'] == i]['Age'].median()

#Create the new age_all field which is initially a direct copy of Age
test_data['Age_All'] = test_data['Age']

#Create the Age_Estimated Flag
test_data['Age_Estimated'] = pd.isnull(test_data['Age']).astype(int)

#Now Update all instances where Age_Estimated flag = 1 with the corresponding Pclass median
for i in pclass_values:
    test_data.loc[(test_data['Age_Estimated'] == 1) & (test_data['Pclass'] == i),'Age_All'] = median_age_pclass[i]
    
#Create the new age_all field which is initially a direct copy of Age
test_data['Fare_All'] = test_data['Fare']

#Now Update all instances where Age_Estimated flag = 1 with the corresponding Pclass median
for i in pclass_values:
    test_data.loc[((test_data['Fare_All'] == 0) | (test_data['Fare_All'].isnull())) & (test_data['Pclass'] == i),'Fare_All'] = median_fare_pclass[i]    
    
#Set all NaN values in Embarked Field = X
test_data.loc[(test_data['Embarked'].isnull()),'Embarked'] = 'X'

#Use mapping dictionary to map values from 'Sex' field to the new field 'Gender'
test_data['Gender'] = test_data['Sex'].map(Sex_Map).astype(int)

#Use mapping dictionary to map values from 'Embarked' field to the new field 'Embarked_From'
test_data['Embarked_From'] = test_data['Embarked'].map(Embarked_Map).astype(int)

#Make the clean dataset and put it into a NumPy array
test = test_data[['Pclass','Age_All','Age_Estimated','Gender','SibSp','Parch','Fare_All','Embarked_From']].values

#Make a table with the passenger ids for use in the output file
PasID = test_data['PassengerId']

#Print top 5 rows of NumPy array to double check the format is correct
print(test[:][0:6])


print(test_data.loc[(test_data['Fare_All'].isnull()),'Fare_All'])
print('Training!')

#Create the random forest object which will include all the parameters
#for the fit
forest = RandomForestClassifier(n_estimators = 1000)

#Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train[0::,1::],train[0::,0])

print('Test!')

#Take the same decision trees and run it on the test data
output = forest.predict(test)

#Print top 5 rows
print(output[:][0:5])
#Create Dataframe with correct structure and using the output and ids arrays
predictions_file = pd.DataFrame({
        "PassengerId": PasID
       ,"Survived"   : output
    })

#Write DataFrame to CSV
predictions_file.to_csv('submission.csv', index=False)

#Print first 5 rows of file and close
print(predictions_file.head())

