# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import csv

import random

numTest = 3000

branchCap = 6

testPointCap = 500

newData = []

branches = []



#print("imported")



associatedBranch = 0



def accuracyCheck(dataList):

    dataPoints = 0

    outliers = 0

    sumPoints = 0

    #print("Datalist is ", str(dataList))

    for branch in dataList:

        for item in branch:

            #print(str(sum), str(item))

            #print('branch is ', str(branch))

            #print('item is ', str(item))

            sumPoints += float(item[8])

        

        if sumPoints > len(branch) - 2:

            sumPoints = len(branch) - sumPoints

        

        outliers += sumPoints

        dataPoints += len(branch)

    

    accuracy = outliers/dataPoints

    return accuracy



def checkBranch(dataPoint, index):

    leftBranch = []

    rightBranch = []

    

    

    for star in range(1, len(trainingData)):

        #print(trainingData[star][index])

        #print(dataPoint)

        if float(trainingData[star][index]) <= float(dataPoint):

            leftBranch += trainingData[star][8]

        else:

            rightBranch += trainingData[star][8]

    

    outliers = 0

    positive = 0

    negative = 0

    for pulsar in leftBranch:

        if int(pulsar) == 1:

            positive += 1

        else:

            negative += 1

    

    #print("leftBranch is " + str(leftBranch))

    

    if positive >= negative:

        outliers += negative

    else:

        outliers += positive

    

    #print("leftGini is", str(leftGini))

    positive = 0

    negative = 0

    

    for pulsar in rightBranch:

        if int(pulsar) == 1:

            positive += 1

        else:

            negative += 1

    

    if len(rightBranch) == 0:

        rightGini = 1

    else:

        if positive >= negative:

            outliers += negative

        else:

            outliers += positive

    

    totalGini = outliers / numTest

    return totalGini

'''

def sortArray(array, index):

    newList = [array[0]]

    rank = 0

    

    for count in range(1, len(array) - 1):

        print(str(rank), str(newList[rank][index]))

        print(str(count), str(array[count][index]))

        

        while newList[rank][index] < array[count][index]:

            rank += 1

            if rank == len(newList):

                break

            

        newList.insert(rank-1, array[count])

    

    return newList

'''

'''

def generateBranches(dataList):

    possibleBranches = []

    

    for dataPoint in range(len(dataList)):

        if dataPoint < len(dataList) - 1: #Check this if error #############################################

            midpoint = dataList[dataPoint] + dataList[dataPoint + 1]

            midpoint = midpoint / 2

            print("midpoint is ", str(midpoint))

            possibleBranches.append(midpoint)

            print(possibleBranches)

    

    return possibleBranches

'''



        

    

        



NUM_OF_EXAMPLES = 17,898



with open('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv', newline='') as f:

    reader = csv.reader(f)

    data = list(reader)

    

testData = []



for count in range(testPointCap):

    dataPoint = data[random.randint(0, len(data) - 1)]

    testData.append(dataPoint)

    data.remove(dataPoint)

    trainingData = data

    

for count in range(len(trainingData)-numTest):

    dataPoint = data[random.randint(0, len(trainingData) - 1)]

    trainingData.remove(dataPoint)

    #newData += dataPoint

    #trainingData = newData

'''

meanProfileSorted = sortArray(trainingData, 0)  

deviationProfileSorted = sortArray(trainingData, 1) 

kurtosisProfileSorted = sortArray(trainingData, 2) 

skewnessProfileSorted = sortArray(trainingData, 3) 

meanCurveSorted = sortArray(trainingData, 4) 

deviationCurveSorted = sortArray(trainingData, 5) 

kurtosisCurveSorted = sortArray(trainingData, 6) 

skewnessCurveSorted = sortArray(trainingData, 7) 

'''

#print("YOUR DATA IS ", str(trainingData))############################################################################

#Should be before

def decisionTree(list):

    trainingData = list

    

    meanProfile = []

    deviationProfile = []

    kurtosisProfile = []

    skewnessProfile = []

    meanCurve = []

    deviationCurve = [] 

    kurtosisCurve = []

    skewnessCurve = []

    

    for star in range(1, len(trainingData)):

        if not star < 1:

            #mlList = trainingData[star]

            meanProfile.append(float(trainingData[star][0]))

            deviationProfile.append(float(trainingData[star][1]))

            #print(trainingData[star])

            #print(trainingData[star][2])

            kurtosisProfile.append(float(trainingData[star][2]))

            skewnessProfile.append(float(trainingData[star][3]))

            meanCurve.append(float(trainingData[star][4]))

            deviationCurve.append(float(trainingData[star][5]))

            kurtosisCurve.append(float(trainingData[star][6]))

            skewnessCurve.append(float(trainingData[star][7]))

    

    #print(meanProfile[0]+meanProfile[1])

    #print(meanProfile)

    #if  type(meanProfile) == list:

    #    print("Good!")

    #else:

    #    print(str(type(meanProfile)))

        

    meanProfile = sorted(meanProfile)

    #print("***", str(meanProfile), "***") #This doesn't output

    deviationProfile = sorted(deviationProfile)

    kurtosisProfile = sorted(kurtosisProfile)

    skewnessProfile = sorted(skewnessProfile)

    meanCurve = sorted(meanCurve)

    deviationCurve = sorted(deviationCurve)

    kurtosisCurve = sorted(kurtosisCurve)

    skewnessCurve = sorted(skewnessCurve)

    

    #print(meanProfile)

    meanProfileBranchCheck = meanProfile #generateBranches(meanProfile) #See function above

    #print(meanProfileBranchCheck)

    deviationProfileBranchCheck = deviationProfile #generateBranches(deviationProfile)

    kurtosisProfileBranchCheck = kurtosisProfile #generateBranches(kurtosisProfile)

    skewnessProfileBranchCheck = skewnessProfile #generateBranches(skewnessProfile)

    meanCurveBranchCheck = meanCurve #generateBranches(meanCurve)

    deviationCurveBranchCheck = deviationCurve #generateBranches(deviationCurve)

    kurtosisCurveBranchCheck = kurtosisCurve #generateBranches(kurtosisCurve)

    skewnessCurveBranchCheck = skewnessCurve #generateBranches(skewnessCurve)

    

    #### BECAUSE IM NOT USING MIDPOINTS, REMEMBER TO USE >= RATHER THAN >

    

    #Above function for checking branches

    

    ########################### IS IT SUPPOSED TO BE CHECK BRANCH 0 THROUGH 7? ###############################################

    

    #print("starting ginis")

    

    ginis = []

    #print("the len(meanProfileBranchCheck) - 1 is "+ str(len(meanProfileBranchCheck) - 1))

    count = 0

    for possibleBranch in meanProfileBranchCheck:

        #print(meanProfileBranchCheck[possibleBranch])

        #print(checkBranch(possibleBranch, 1))

        #print([meanProfileBranchCheck[count], checkBranch(possibleBranch, 0), "meanProfile"])

        ginis.append([meanProfileBranchCheck[count], checkBranch(possibleBranch, 0), 0]) #Data attribute

        count += 1

    

    #print("meanProfile", str(len(meanProfile)), str(count))

    

    count = 0

    for possibleBranch in deviationProfileBranchCheck:

        ginis.append([deviationProfileBranchCheck[count], checkBranch(possibleBranch, 1), 1])

        count += 1

    

    #print("deviationProfile", str(len(deviationProfile)), str(count))

    count = 0

    for possibleBranch in kurtosisProfileBranchCheck:

        ginis.append([kurtosisProfileBranchCheck[count], checkBranch(possibleBranch, 2), 2])

        count += 1

        

    count = 0

    for possibleBranch in skewnessProfileBranchCheck:

        ginis.append([skewnessProfileBranchCheck[count], checkBranch(possibleBranch, 3), 3])

        count += 1

        

    count = 0

    #print(ginis)

    #should be branch, type, gini

    for possibleBranch in meanCurveBranchCheck:

        ginis.append([meanCurveBranchCheck[count], checkBranch(possibleBranch, 4), 4]) #Data attribute

        count += 1

        

    count = 0

    for possibleBranch in deviationCurveBranchCheck:

        ginis.append([deviationCurveBranchCheck[count], checkBranch(possibleBranch, 5), 5])

        count += 1

        

    count = 0

    for possibleBranch in kurtosisCurveBranchCheck:

        ginis.append([kurtosisCurveBranchCheck[count], checkBranch(possibleBranch, 6), 6])

        count += 1

        

    count = 0

    for possibleBranch in skewnessCurveBranchCheck:

        ginis.append([skewnessCurveBranchCheck[count], checkBranch(possibleBranch, 7), 7])

        count += 1



    #print(ginis)###

    #to find lowest gini

    currentLowestGini = 1

    associatedBranch = []

    

    for gini in ginis:

        

        if gini[1] < currentLowestGini:

            currentLowestGini = gini[1]

            associatedBranch = gini

    

    

    ginis.remove(associatedBranch)

    branches.append(associatedBranch)

    

            #print(gini)

    #print(type(associatedBranch))###

    #print("ginis is", str(ginis))

    #print("associated branch is ", str(associatedBranch))###

    

    leftBranch = []

    rightBranch = []

    newList = []

    #print(trainingData)

    for item in trainingData:



        if float(item[associatedBranch[2]]) < associatedBranch[0]:

            leftBranch.append(item)

        else:

            rightBranch.append(item)

    

    newList.append(leftBranch)

    newList.append(rightBranch)

    #print("*"*40)

    #print(newList)

    trainingData = newList

    topGinis = []

    #print(gini)

    

    

    for count in range(branchCap):

        currentLowestGini = 1

        

        for gini in ginis:

        

            if gini[1] < currentLowestGini:

                currentLowestGini = gini[1]

                associatedBranch = gini

        #print(gini)

        ginis.remove(associatedBranch)

        topGinis.append(associatedBranch)

    

    ginis = topGinis

        

    ############################# SSSSSSSSSSSSSSSSOOOOOOOOOOORRRRRRTTTT GINIS ##########################################

    count = 1

    while not accuracyCheck(trainingData) > 0.8 and count < branchCap:

        #print("count = ", str(count))

        leftBranch = []

        rightBranch = []

        newList = []

        

        for item in trainingData:

            for List in item:

                #print(ginis[count][2])

                #print(str(List[ginis[count][2]]), str(ginis[count][0]))

                if float(List[ginis[count][2]]) <= ginis[count][0]:

                    leftBranch.append(List)

                else:

                    rightBranch.append(List)

                    

            newList.append(leftBranch)

            newList.append(rightBranch)

        

        branches.append(ginis[count])

        trainingData = newList

        count += 1

    

    

    print("Your program worked. It had an accuracy of ", str(1 - accuracyCheck(trainingData)))

    

    endResult = []

    for branch in trainingData:

        points = 0

        hits = 0

        

        for point in branch:

            #if point[ginis[0][2]] <

            hits += int(point[8])

            points += 1

        

        if hits * 2 >= points:

            endResult.append(1)

        else:

            endResult.append(0)

        

    pointsTested = 0

    correct = 0

    pulsarsCorrect = 0

    pulsarSum = 0

    

    for point in testData:

        pulsarSum += int(point[8])

        

        resultIndex = 0

        for count in range(len(ginis)):

            resultIndex *= 2

            #print(count)

            #print(ginis[count][2])

            #print(ginis[count])

            #print(point)

            if float(point[ginis[count][2]]) > ginis[count][0]:

                resultIndex += 1

        

        #print(endResult)

        #print(len(endResult))

        #print(resultIndex)

        

        if endResult[resultIndex] == int(point[8]):

            if endResult[resultIndex] == 1:

                pulsarsCorrect += 1

            correct += 1



        pointsTested += 1

    

    accuracy = correct / pointsTested

    print("The tested accuracy was ", str(accuracy))

    print("There were ", str(pulsarSum), " pulsars in the test set")

    print(str(pulsarsCorrect), " pulsars were identified correctly")

    

    '''

    point = []

    for number in range(8):

        attribute = float(input("Attribute: "))

        point.append(attribute)

    

    resultIndex = 0

    for count in range(len(ginis)):

        resultIndex *= 2

        #print(count)

        #print(ginis[count][2])

        #print(ginis[count])

        #print(point)

        if float(point[ginis[count][2]]) > ginis[count][0]:

            resultIndex += 1

        

    print(endResult[resultIndex])

    '''

#print(len(testData))

#print(testData)

#print("########" * 100)

#print(data)

    

decisionTree(trainingData)

#print(type(associatedBranch))###





#print(len(testData))

#print(testData)

#print("########" * 100)

#print(data)



# 99.3671875	41.57220208	1.547196967	4.154106043	27.55518395	61.71901588	2.20880796	3.662680136

#print(data) 
