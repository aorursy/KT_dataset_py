##################### IMPORTING ALL LIBRARIES & LOADING THE DATASET

import pandas as pd

import numpy as np

import matplotlib as plt

import random

df = pd.read_csv('Accused-Witches-Data-Set.csv')
######################## N. of witches per residence

%matplotlib inline

df[' Residence '].value_counts().plot(kind='barh')
######################### N. of witches accused per month of 1692

#removing the negative values and replacing them with 1

accusationTimes = df['Month of Accusation'].replace(-1, 1)

#counting the number of accusations per month

accusationTimes = list(accusationTimes)

cleanedValues = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0}

for x in accusationTimes:

    cleanedValues[str(x)] += 1

#creating a labels list for the chart

lblList = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug','Sep','Oct','Nov']

#creating a colors list to user for the chart

clrsList = ['c', 'm', 'r', 'b', 'g', 'k', 'w', '#f8013b', '#ccff00', '#e908f1', '#31698a']

#plotting the pie chart itself with settings for the labels,colors,angle,percentages and shadow

plt.pyplot.pie(list(cleanedValues), labels = lblList, colors = clrsList, startangle = 90, autopct='%1.1f%%', shadow = True )
######################################### Residence - Month of Accusation Correlation

#creating a list of months for labels

monthsList = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug','Sep','Oct','Nov']

#a list for the different settlements

residenceList = []

for x in df[' Residence ']:

    if x not in residenceList:

        residenceList.append(x)

#sorting the list alphabetically

residenceList = sorted(residenceList)

#now onto counting how many women were accused of being witches in each settlement per month

residenceWitchesPerMonth = {}

for x in residenceList:

    residenceWitchesPerMonth[x] = [0,0,0,0,0,0,0,0,0,0,0]

listNumCounter = 0

while(listNumCounter < 152):

    tempList = [x for x in df.loc[listNumCounter]]

    listNumCounter += 1

    residenceWitchesPerMonth[tempList[1]][tempList[2]-1] += 1

#making a new colors list with 24 different random rgb values for each settlement

clrsList2 = []

z = 0

while(z < 24):

    clrsList2.append((random.random(), random.random(),random.random()))

    z += 1

#plotting + plotting settings such as labels (1 for every settlement) and colors

plt.pyplot.xlabel(monthsList)

for i in range(0,24):

    plt.pyplot.plot([],[], color=clrsList2[i], label=residenceList[i], linewidth=5)

plt.pyplot.legend()

plt.pyplot.stackplot([x for x in range(0,11)], residenceWitchesPerMonth[residenceList[0]], residenceWitchesPerMonth[residenceList[1]], residenceWitchesPerMonth[residenceList[2]], residenceWitchesPerMonth[residenceList[3]], residenceWitchesPerMonth[residenceList[4]], residenceWitchesPerMonth[residenceList[5]], residenceWitchesPerMonth[residenceList[6]], residenceWitchesPerMonth[residenceList[7]],residenceWitchesPerMonth[residenceList[8]], residenceWitchesPerMonth[residenceList[9]], residenceWitchesPerMonth[residenceList[10]], residenceWitchesPerMonth[residenceList[11]], residenceWitchesPerMonth[residenceList[12]], residenceWitchesPerMonth[residenceList[13]], residenceWitchesPerMonth[residenceList[14]], residenceWitchesPerMonth[residenceList[15]], residenceWitchesPerMonth[residenceList[16]], residenceWitchesPerMonth[residenceList[17]], residenceWitchesPerMonth[residenceList[18]], residenceWitchesPerMonth[residenceList[19]], residenceWitchesPerMonth[residenceList[20]], residenceWitchesPerMonth[residenceList[21]], residenceWitchesPerMonth[residenceList[22]], residenceWitchesPerMonth[residenceList[23]], colors=clrsList2)