import csv
import random

print("imported")

associatedBranch = 0
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
def checkBranch(dataPoint, index):
    leftBranch = []
    rightBranch = []
    
    
    for star in range(1, len(trainingData)):
        print(trainingData[star][index])
        print(dataPoint)
        if float(trainingData[star][index]) <= float(dataPoint):
            leftBranch += trainingData[star][8]
        else:
            rightBranch += trainingData[star][8]
        
    positive = 0
    negative = 0
    for pulsar in leftBranch:
        if pulsar == 1:
            positive += 1
        else:
            negative += 1
    
    print("leftBranch is " + str(leftBranch))
    
    if positive >= negative:
        leftGini = negative / len(leftBranch)
    else:
        leftGini = positive / len(leftBranch)
        
    positive = 0
    negative = 0
    for pulsar in rightBranch:
        if pulsar == 1:
            positive += 1
        else:
            negative += 1
    
    if positive >= negative:
        rightGini = negative / len(rightBranch)
    else:
        rightGini = positive / len(rightBranch)
    
    totalGini = leftGini + rightGini
    return totalGini
        
    
        

NUM_OF_EXAMPLES = 17,898

with open('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
testData = []

for count in range(4480):
    dataPoint = data[random.randint(0, len(data) - 1)]
    testData += dataPoint
    data.remove(dataPoint)
    trainingData = data

#Should be before
def decisionTree(list):
    meanProfile = []
    deviationProfile = []
    kurtosisProfile = []
    skewnessProfile = []
    meanCurve = []
    deviationCurve = [] 
    kurtosisCurve = []
    skewnessCurve = []
    
    #print(trainingData)
    for star in range(1, len(trainingData)):
        #mlList = trainingData[star]
        meanProfile.append(float(trainingData[star][0]))
        deviationProfile.append(float(trainingData[star][1]))
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
    
    print("starting ginis")
    
    ginis = []
    print("the len(meanProfileBranchCheck) - 1 is "+ str(len(meanProfileBranchCheck) - 1))
    count = 0
    for possibleBranch in meanProfileBranchCheck:
        #print(meanProfileBranchCheck[possibleBranch])
        #print(checkBranch(possibleBranch, 1))
        print([meanProfileBranchCheck[count], checkBranch(possibleBranch, 0), "meanProfile"])
        ginis += [meanProfileBranchCheck[count], checkBranch(possibleBranch, 0), "meanProfile"] #Data attribute
        count += 1
    
    print("meanProfile", str(len(meanProfile)), str(count))
    
    '''
    count = 0
    for possibleBranch in deviationProfileBranchCheck:
        ginis += [deviationProfileBranchCheck[count], checkBranch(possibleBranch, 1), "deviationProfile"]
        count += 1
    
    print("deviationProfile", str(len(deviationProfile)), str(count))
    count = 0
    for possibleBranch in kurtosisProfileBranchCheck:
        ginis += [kurtosisProfileBranchCheck[count], checkBranch(possibleBranch, 2), "kurtosisProfile"]
        count += 1
        
    count = 0
    for possibleBranch in skewnessProfileBranchCheck:
        ginis += [skewnessProfileBranchCheck[count], checkBranch(possibleBranch, 3), "skewnessProfile"]
        count += 1
        
    count = 0
    #print(ginis)
    #should be branch, type, gini
    for possibleBranch in meanCurveBranchCheck:
        ginis += [meanCurveBranchCheck[count], checkBranch(possibleBranch, 4), "meanCurve"] #Data attribute
        count += 1
        
    count = 0
    for possibleBranch in deviationCurveBranchCheck:
        ginis += [deviationCurveBranchCheck[count], checkBranch(possibleBranch, 5), "deviationCurve"]
        count += 1
        
    count = 0
    for possibleBranch in kurtosisCurveBranchCheck:
        ginis += [kurtosisCurveBranchCheck[count], checkBranch(possibleBranch, 6), "kurtosisCurve"]
        count += 1
        
    count = 0
    for possibleBranch in skewnessCurveBranchCheck:
        ginis += [skewnessCurveBranchCheck[count], checkBranch(possibleBranch, 7), "skewnessCurve"]
        count += 1
    '''
    print("ginis done")

    #print(ginis)###
    #to find lowest gini
    currentLowestGini = 1
    associatedBranch = []
    
    for gini in ginis:
        #print(gini)###
        if gini[1] < currentLowestGini:
            currentLowestGini = gini[1]
            associatedBranch = gini
            #print(gini)
    
    
decisionTree(trainingData)
#print(type(associatedBranch))###
print(associatedBranch)###

#print(len(testData))
#print(testData)
#print("########" * 100)
#print(data)

    
#print(data)