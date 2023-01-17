# Import

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



## Def functions for preprocessing

def strSplit(string,splitChar,idx):

    # Split strings on character, return requested index

    string = string.split(splitChar)

    return string[idx]

    

def numericPeople(data):

    # Make people numeric

    data['people_id'] = data['people_id'].apply(strSplit, splitChar='_', idx=1)

    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)    

    return data

    

def ppActs(data):



    # Drop outcome and return in sperate vector

    if 'outcome' in data.columns:    

        outcome = data['outcome']

        data = data.drop('outcome', axis=1)

    else:

        outcome = 0

    # Drop activity ID

    data = data.drop(['date', 'activity_id'], axis=1)

        

    # Make people numeric

    data = numericPeople(data)

    

    # Convert rest to numeric

    for c in data.columns:

        data[c] = data[c].fillna('type 0')

        if type(data[c][1]) == str:        

           data[c] = data[c].apply(strSplit, splitChar=' ', idx=1)



    return data, outcome



def ppPeople(data):

    

    # Drop date    

    data = data.drop('date', axis=1)

    

    # Make people numeric

    data = numericPeople(data)      

    

    for c in data.columns:

        if type(data[c][1]) == np.bool_:

            data[c] = pd.to_numeric(data[c]).astype(int)

        elif type(data[c][1]) == str:

            data[c] = data[c].apply(strSplit, splitChar=' ', idx=1)

    

    return data



## Import data

# Just training data with known outcomes and people

actTrain = pd.read_csv('../input/act_train.csv')

people = pd.read_csv('../input/people.csv')



## Preprocess 

XTrain, YTrain = ppActs(actTrain)

proPeople = ppPeople(people)    



# Merge in people

XTrain = XTrain.merge(proPeople, how='left', on='people_id')
# Unique people

# Total

nTotalUnq = len(proPeople['people_id'].unique())

# Total people present in Training set

peopleTrain = XTrain['people_id'].unique()

nTrainUnq = len(peopleTrain)



print('In total there are', nTotalUnq, 'people, of which', 

      nTrainUnq, '(', round(nTrainUnq/nTotalUnq*100), '%) are present in the training set.')
# All activities

vcAll = proPeople['people_id'].value_counts()

# Activities in training set

vcTrain = XTrain['people_id'].value_counts()

print('In total', len(XTrain), 'activities have been completed by these', nTrainUnq, 'people')
# Overall success rate

vcS = XTrain[YTrain==1]['people_id'].value_counts()

vcF = XTrain[YTrain==0]['people_id'].value_counts()



# Overall success rate in training set

successRate = sum(YTrain==1) / len(YTrain)

print('Overall', sum(YTrain==1), '/', sum(YTrain==0), 

'(', round(successRate*100), '%) activities in the training set were successful')
# Success rate by person

vcSF = pd.concat([vcS,vcF], axis=1, keys = ['Success', 'Fail'])

vcSF[vcSF.isnull()] = 0

vcSF['SuccessRate'] = vcSF['Success'] / (vcSF['Success']+vcSF['Fail'])



# Best people

# Top 10

print('These are the best people')

vcSF = vcSF.sort_values(by=['SuccessRate', 'Success'], 

                        ascending=[False, False])

print(vcSF.iloc[0:10,:])

vcSF.iloc[0:10,:].plot(kind="bar")

plt.ylabel('Successful activity count')

plt.title('Top 10')

fig = plt.gcf()

# Bottom 10

print('These are the worst people:')

vcSF = vcSF.sort_values(by=['SuccessRate', 'Fail'], 

                        ascending=[True, False])

print(vcSF.iloc[0:10,:])

vcSF.iloc[0:10,:].plot(kind="bar")

plt.ylabel('Failed activity count:')

plt.title('Bottom 10')

fig = plt.gcf()
n, bins, patches = plt.hist(vcSF['SuccessRate'], 5, facecolor='green', alpha=0.75)

plt.ylabel('People count')

plt.xlabel('SuccessRate')
# Activities completed

nAcs = XTrain['activity_category'].value_counts()

print('Numbers of each kind of activity done:')



nAcs.plot(kind='bar')

plt.ylabel('Count')

plt.xlabel('Activity type')

plt.title('Activities completed')

print(nAcs)
# Success rate by activity

acS = XTrain[YTrain==1]['activity_category'].value_counts()

acF = XTrain[YTrain==0]['activity_category'].value_counts()



acSF = pd.concat([acS,acF], axis=1, keys = ['Success', 'Fail'])

acSF[vcSF.isnull()] = 0

acSF['SuccessRate'] = acSF['Success'] / (acSF['Success']+acSF['Fail'])



# Bargraph of successRate

acSF['SuccessRate'].plot(kind='bar')

plt.ylabel('Success rate')

plt.xlabel('Activity type')



# And Success and failure counts

acSF.plot(kind='bar')

plt.ylabel('Count')

plt.xlabel('Activity type')



print(acSF)