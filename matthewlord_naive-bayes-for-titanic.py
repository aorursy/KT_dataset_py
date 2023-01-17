# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from matplotlib.pyplot import plot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



# Load in data and have a peek at the training data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data= pd.read_csv("/kaggle/input/titanic/test.csv")



train_data.head(-5)
# Calculate total number to get a feel for the data

total_survived = train_data.loc[train_data['Survived'] == 1].count()

n_survived = total_survived['Survived']

print("Total Survived: ", n_survived)



total_dead = train_data.loc[train_data['Survived'] == 0].count()

n_deceased = total_dead['Survived']

print("Total Deceased: ", n_deceased)



n = train_data.shape[0]

print("Total Num: ", n)
# Calculate Proportions

print("********* Proportions ***********")



p_Total_Survived = n_survived / n

p_Total_Deceased = n_deceased / n 

print("Proportion Survived: ", p_Total_Survived)

print("Proportion Deceased: ", p_Total_Deceased)
# Create method to calculate likelihood of survival for each feature conditional on given survival status, to be used in the numerator of the Bayes formula later.

def calcConditionalProbForFeature(feature, featureValue, survived):

    # Get count of rows that have the feature match the value being tested along with given survival outcome

    sample = train_data.loc[(train_data[feature] == featureValue) & (train_data['Survived'] == survived)].shape[0]

    pr = sample / n

    # print("P(" + str(featureValue) + "|Survived: " + str(survived) + "): ", pr)

    return pr



# Test this, should add to 1 as all genders/outcomes are covered.

print(calcConditionalProbForFeature("Sex", "male", 0) + calcConditionalProbForFeature("Sex", "male", 1) + calcConditionalProbForFeature("Sex", "female", 1) + calcConditionalProbForFeature("Sex", "female", 0))
# Create method to calculate likelihood of survival for each feature in total, to be used in the denominator of the Bayes formula later.

def calcTotalProbForFeature(feature, featureValue):

    # Get count of rows that have the feature match the value being tested along with given survival outcome

    sample = train_data.loc[(train_data[feature] == featureValue)].shape[0]

    pr = sample / n

    # print("P(" + str(featureValue) + "|Survived: " + str(survived) + "): ", pr)

    return pr



#Edit the Age into brackets to turn the data from continuous into discrete



def createAgeBrackets(data):



    #Fill N/As

    data.Age.fillna(data.Age.mean(), inplace=True)

    

    #Use pd.cut to create the age brackets

    data['AgeBrackets'] = pd.cut(x=train_data['Age'], bins=[-1,3,7,13,19,29,39,49,59,69,99])



    #Get total age count

    totalAgeCounts = data['AgeBrackets'].value_counts()



    # Calculate the survival proportion for each age

    survivedOnly = data.loc[train_data['Survived'] == 1]

    survivedOnlyAgeCounts = survivedOnly['AgeBrackets'].value_counts()

    survivalAgeProp = survivedOnlyAgeCounts / totalAgeCounts



    #Plot this out into a bar so we get a rough idea

    survivalAgeProp.plot.bar()



    return data
#Edit the Fares into brackets as well

def createFareBrackets(data):

    #Use pd.cut to create the Fare brackets

    data['FareBrackets'] = pd.cut(x=train_data['Fare'], bins=[-1,8,12,16,20,30,40,50,60,70,80,90,999])

    val_counts = data['FareBrackets'].value_counts()

    

    # Calculate the survival proportion for each Fare

    survivedOnly = data.loc[train_data['Survived'] == 1]

    survivedOnlyAgeCounts = survivedOnly['FareBrackets'].value_counts()

    survivalAgeProp = survivedOnlyAgeCounts / val_counts

    

    #Remove any NaNs which occur by dividing ages in which 0 people survived

    survivalAgeProp = survivalAgeProp.fillna(0)



    #Plot this out into a bar so we get a rough idea

    survivalAgeProp.plot.bar()

    

    return data
#  This will calculate the probability of a survival outcome for the given feature and return it in a series

def calcNumeratorForCol(data, feature, survived):

    output = [calcConditionalProbForFeature(feature, value, survived) for value in data[feature]]

    series = pd.Series(output)

    return series



def calcDenominatorforCol(data, feature):

    output = [calcTotalProbForFeature(feature, value) for value in data[feature]]

    series = pd.Series(output)

    return series



# This compares the likelihood of surviving the event with given features vs deceased, the model will then predict whatever is most likely.

def calcSurvivalRate(data):

    #Calc survival chances

    # Numerator

    probAgeSurvive = calcNumeratorForCol(data, 'AgeBrackets', 1)

    probSexSurvive = calcNumeratorForCol(data, 'Sex', 1)

    probClassSurvive = calcNumeratorForCol(data, 'Pclass', 1)

    #probFareSurvive = calcNumeratorForCol(data, 'FareBrackets', 1)

    combinedSurviveNumerator = probAgeSurvive * probSexSurvive * probClassSurvive * p_Total_Survived

    

    #Calc deceased chances

    # Numerator

    probAgeDecease = calcNumeratorForCol(data, 'AgeBrackets', 0)

    probSexDecease = calcNumeratorForCol(data, 'Sex', 0)

    probClassDecease = calcNumeratorForCol(data, 'Pclass', 0)

    #probFareDecease = calcNumeratorForCol(data, 'FareBrackets', 0)

    combinedDeceasedNumerator = probSexDecease * probAgeDecease * probClassDecease * p_Total_Deceased

    

    #Denominators

    probAge = calcDenominatorforCol(data, 'AgeBrackets')

    probSex = calcDenominatorforCol(data, 'Sex')

    probClass = calcDenominatorforCol(data, 'Pclass')

    #probFare = calcDenominatorforCol(data, 'FareBrackets')

    combinedDenominator = probAge * probSex * probClass

    

    # Calculate using Bayes Thereom

    BayesSurvive = combinedSurviveNumerator/combinedDenominator

    BayesDeceased = combinedDeceasedNumerator/combinedDenominator

    



    

    #Pick whatever outcome is more likely and convert to submission format (1/0 instead of true/false)

    data['Predictions'] = BayesSurvive > BayesDeceased

    data.Predictions = data.Predictions.astype(int)

    return data



# Format train and test data

train_data = createAgeBrackets(train_data)

test_data = createAgeBrackets(test_data)



#train_data = createFareBrackets(train_data)

#test_data = createFareBrackets(test_data)



#predict

predictions = calcSurvivalRate(test_data)



output = pd.DataFrame({'PassengerId': predictions.PassengerId, 'Survived': predictions['Predictions']})

output.to_csv('my_submission.csv', index=False)