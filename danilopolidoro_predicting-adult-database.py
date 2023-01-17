# imports and settings

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import threading
from sklearn import neighbors, tree, ensemble
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import AdaBoostClassifier

%config InlineBackend.figure_format = 'svg'
mpl.rcParams['figure.dpi']= 300
# loading databases
adult = pd.read_csv('../input/atividade-4-versao-1/train_data.csv',names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?", skiprows = 1).dropna()

adultTest = pd.read_csv('../input/test-data-with-targer/test_data.csv',names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?", skiprows = 1).dropna()
adult.shape
adult.head()
ageData = adult['Age'].value_counts().to_dict()
xAxis = [key for key in ageData]
yAxis = [ageData[key] for key in ageData]
plt.scatter(xAxis, yAxis)
plt.xlabel('Age')
plt.ylabel('Amount of people');
ageTargetData = {}
for index, element in enumerate(adult['Age']):
    if index in adult['Target']:
        if element not in ageTargetData:
            if adult['Target'][index] == '<=50K':
                ageTargetData[element] = [1,0]
            else:
                ageTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                ageTargetData[element][0] +=1
            else:
                ageTargetData[element][1] += 1

xAxis = [key for key in ageTargetData]
y1Axis = [100*ageTargetData[key][0]/(ageTargetData[key][0]+ageTargetData[key][1]) for key in ageTargetData]
y2Axis = [100*ageTargetData[key][1]/(ageTargetData[key][0]+ageTargetData[key][1]) for key in ageTargetData]


plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.legend()

plt.xlabel('Age')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
adult['Workclass'].value_counts().plot(kind='bar')
plt.xlabel('Work Class')
plt.ylabel('Amount of people');
workclassTargetData = {}
for index, element in enumerate(adult['Workclass']):
    if index in adult['Target']:
        if element not in workclassTargetData:
            if adult['Target'][index] == '<=50K':
                workclassTargetData[element] = [1,0]
            else:
                workclassTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                workclassTargetData[element][0] +=1
            else:
                workclassTargetData[element][1] += 1

            
xAxis = [key for key in workclassTargetData]
y1Axis = [workclassTargetData[key][0]/(workclassTargetData[key][0]+workclassTargetData[key][1]) for key in workclassTargetData]
y2Axis = [workclassTargetData[key][1]/(workclassTargetData[key][0]+workclassTargetData[key][1]) for key in workclassTargetData]

plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('Workclass')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
fnList0 = adult['fnlwgt'].tolist()
fnList1 = [el/1000 for el in fnList0]
plt.hist(fnList1)
plt.xlabel('fnlwgt $[\\times10^4]$')
plt.ylabel('Absolut Frequency');
fnTargetData = {}
for index, element in enumerate(adult['fnlwgt']):
    if index in adult['Target']:
        if element not in fnTargetData:
            if adult['Target'][index] == '<=50K':
                fnTargetData[element] = [1,0]
            else:
                fnTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                fnTargetData[element][0] +=1
            else:
                fnTargetData[element][1] += 1

            
xAxis = [key/1000 for key in fnTargetData]
y1Axis = [fnTargetData[key][0]/(fnTargetData[key][0]+fnTargetData[key][1]) for key in fnTargetData]
y2Axis = [fnTargetData[key][1]/(fnTargetData[key][0]+fnTargetData[key][1]) for key in fnTargetData]

%config InlineBackend.figure_format = 'png'
plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('fnlwgt $[ \\times10^4]$')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
%config InlineBackend.figure_format = 'svg'
adult['Education'].value_counts().plot(kind='bar')
plt.xlabel('Education')
plt.ylabel('Amount of people');
educationTargetData = {}
for index, element in enumerate(adult['Education']):
    if index in adult['Target']:
        if element not in educationTargetData:
            if adult['Target'][index] == '<=50K':
                educationTargetData[element] = [1,0]
            else:
                educationTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                educationTargetData[element][0] +=1
            else:
                educationTargetData[element][1] += 1

            
xAxis = [key for key in educationTargetData]
y1Axis = [educationTargetData[key][0]/(educationTargetData[key][0]+educationTargetData[key][1]) for key in educationTargetData]
y2Axis = [educationTargetData[key][1]/(educationTargetData[key][0]+educationTargetData[key][1]) for key in educationTargetData]

plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('Education')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
educationTargetData = {}
for index, element in enumerate(adult['Education-Num']):
    if index in adult['Target']:
        if element not in educationTargetData:
            if adult['Target'][index] == '<=50K':
                educationTargetData[element] = [1,0]
            else:
                educationTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                educationTargetData[element][0] +=1
            else:
                educationTargetData[element][1] += 1

            
xAxis = [key for key in educationTargetData]
y1Axis = [educationTargetData[key][0]/(educationTargetData[key][0]+educationTargetData[key][1]) for key in educationTargetData]
y2Axis = [educationTargetData[key][1]/(educationTargetData[key][0]+educationTargetData[key][1]) for key in educationTargetData]

plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('Education')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
adult['Marital Status'].value_counts().plot(kind='bar')
plt.xlabel('Marital Status')
plt.ylabel('Amount of people');
marriageTargetData = {}
for index, element in enumerate(adult['Marital Status']):
    if index in adult['Target']:
        if element not in marriageTargetData:
            if adult['Target'][index] == '<=50K':
                marriageTargetData[element] = [1,0]
            else:
                marriageTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                marriageTargetData[element][0] +=1
            else:
                marriageTargetData[element][1] += 1

            
xAxis = [key for key in marriageTargetData]
y1Axis = [marriageTargetData[key][0]/(marriageTargetData[key][0]+marriageTargetData[key][1]) for key in marriageTargetData]
y2Axis = [marriageTargetData[key][1]/(marriageTargetData[key][0]+marriageTargetData[key][1]) for key in marriageTargetData]

plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('Marital Status')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
adult['Occupation'].value_counts().plot(kind='bar')
plt.xlabel('Occupation')
plt.ylabel('Amount of people');
occupationTargetData = {}
for index, element in enumerate(adult['Occupation']):
    if index in adult['Target']:
        if element not in occupationTargetData:
            if adult['Target'][index] == '<=50K':
                occupationTargetData[element] = [1,0]
            else:
                occupationTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                occupationTargetData[element][0] +=1
            else:
                occupationTargetData[element][1] += 1

            
xAxis = [key for key in occupationTargetData]
y1Axis = [occupationTargetData[key][0]/(occupationTargetData[key][0]+occupationTargetData[key][1]) for key in occupationTargetData]
y2Axis = [occupationTargetData[key][1]/(occupationTargetData[key][0]+occupationTargetData[key][1]) for key in occupationTargetData]

plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('Occupation')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
adult['Relationship'].value_counts().plot(kind='bar')
plt.xlabel('Relationship')
plt.ylabel('Amount of people');
relationshipTargetData = {}
for index, element in enumerate(adult['Relationship']):
    if index in adult['Target']:
        if element not in relationshipTargetData:
            if adult['Target'][index] == '<=50K':
                relationshipTargetData[element] = [1,0]
            else:
                relationshipTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                relationshipTargetData[element][0] +=1
            else:
                relationshipTargetData[element][1] += 1

            
xAxis = [key for key in relationshipTargetData]
y1Axis = [relationshipTargetData[key][0]/(relationshipTargetData[key][0]+relationshipTargetData[key][1]) for key in relationshipTargetData]
y2Axis = [relationshipTargetData[key][1]/(relationshipTargetData[key][0]+relationshipTargetData[key][1]) for key in relationshipTargetData]

plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('Relationship')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
adult['Race'].value_counts().plot(kind='bar')
plt.xlabel('Race')
plt.ylabel('Amount of people');
raceTargetData = {}
for index, element in enumerate(adult['Race']):
    if index in adult['Target']:
        if element not in raceTargetData:
            if adult['Target'][index] == '<=50K':
                raceTargetData[element] = [1,0]
            else:
                raceTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                raceTargetData[element][0] +=1
            else:
                raceTargetData[element][1] += 1

            
xAxis = [key for key in raceTargetData]
y1Axis = [raceTargetData[key][0]/(raceTargetData[key][0]+raceTargetData[key][1]) for key in raceTargetData]
y2Axis = [raceTargetData[key][1]/(raceTargetData[key][0]+raceTargetData[key][1]) for key in raceTargetData]

plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('Race')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
adult['Sex'].value_counts().plot(kind='pie')
plt.xlabel('Sex')
plt.ylabel('Amount of people');
sexTargetData = {}
for index, element in enumerate(adult['Sex']):
    if index in adult['Target']:
        if element not in sexTargetData:
            if adult['Target'][index] == '<=50K':
                sexTargetData[element] = [1,0]
            else:
                sexTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                sexTargetData[element][0] +=1
            else:
                sexTargetData[element][1] += 1

            
xAxis = [key for key in sexTargetData]
y1Axis = [sexTargetData[key][0]/(sexTargetData[key][0]+sexTargetData[key][1]) for key in sexTargetData]
y2Axis = [sexTargetData[key][1]/(sexTargetData[key][0]+sexTargetData[key][1]) for key in sexTargetData]

plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('Sex')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
gainData = adult['Capital Gain'].value_counts().to_dict()
xAxis = [key for key in gainData]
yAxis = [gainData[key] for key in gainData]
plt.scatter(xAxis, yAxis)
plt.xlabel('Capital Gain')
plt.ylabel('Amount of people');
capitalTargetData = {}
for index, element in enumerate(adult['Capital Gain']):
    if index in adult['Target']:
        if element not in capitalTargetData:
            if adult['Target'][index] == '<=50K':
                capitalTargetData[element] = [1,0]
            else:
                capitalTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                capitalTargetData[element][0] +=1
            else:
                capitalTargetData[element][1] += 1

xAxis = [key for key in capitalTargetData]
y1Axis = [100*capitalTargetData[key][0]/(capitalTargetData[key][0]+capitalTargetData[key][1]) for key in capitalTargetData]
y2Axis = [100*capitalTargetData[key][1]/(capitalTargetData[key][0]+capitalTargetData[key][1]) for key in capitalTargetData]


plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.legend()

plt.xlabel('Capital Gain')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
lossData = adult['Capital Loss'].value_counts().to_dict()
xAxis = [key for key in lossData]
yAxis = [lossData[key] for key in lossData]
plt.scatter(xAxis, yAxis)
plt.xlabel('Capital Loss')
plt.ylabel('Amount of people');
capitalTargetData = {}
for index, element in enumerate(adult['Capital Loss']):
    if index in adult['Target']:
        if element not in capitalTargetData:
            if adult['Target'][index] == '<=50K':
                capitalTargetData[element] = [1,0]
            else:
                capitalTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                capitalTargetData[element][0] +=1
            else:
                capitalTargetData[element][1] += 1

xAxis = [key for key in capitalTargetData]
y1Axis = [100*capitalTargetData[key][0]/(capitalTargetData[key][0]+capitalTargetData[key][1]) for key in capitalTargetData]
y2Axis = [100*capitalTargetData[key][1]/(capitalTargetData[key][0]+capitalTargetData[key][1]) for key in capitalTargetData]


plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.legend()

plt.xlabel('Capital Loss')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
hoursData = adult['Hours per week'].value_counts().to_dict()
xAxis = [key for key in hoursData]
yAxis = [hoursData[key] for key in hoursData]
plt.scatter(xAxis, yAxis)
plt.xlabel('Hours per week')
plt.ylabel('Amount of people');
hoursTargetData = {}
for index, element in enumerate(adult['Hours per week']):
    if index in adult['Target']:
        if element not in hoursTargetData:
            if adult['Target'][index] == '<=50K':
                hoursTargetData[element] = [1,0]
            else:
                hoursTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                hoursTargetData[element][0] +=1
            else:
                hoursTargetData[element][1] += 1

xAxis = [key for key in hoursTargetData]
y1Axis = [100*hoursTargetData[key][0]/(hoursTargetData[key][0]+hoursTargetData[key][1]) for key in hoursTargetData]
y2Axis = [100*hoursTargetData[key][1]/(hoursTargetData[key][0]+hoursTargetData[key][1]) for key in hoursTargetData]


plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.legend()

plt.xlabel('Hours per week')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
adult['Country'].value_counts().plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Amount of people');
countryTargetData = {}
for index, element in enumerate(adult['Country']):
    if index in adult['Target']:
        if element not in countryTargetData:
            if adult['Target'][index] == '<=50K':
                countryTargetData[element] = [1,0]
            else:
                countryTargetData[element] = [0,1]
        else:
            if adult['Target'][index] == '<=50K':
                countryTargetData[element][0] +=1
            else:
                countryTargetData[element][1] += 1

xAxis = [key for key in countryTargetData]
y1Axis = [100*countryTargetData[key][0]/(countryTargetData[key][0]+countryTargetData[key][1]) for key in countryTargetData]
y2Axis = [100*countryTargetData[key][1]/(countryTargetData[key][0]+countryTargetData[key][1]) for key in countryTargetData]


plt.scatter(xAxis, y1Axis, label = '<=50K', color = 'red')
plt.scatter(xAxis, y2Axis, label = '>50K', color = 'green')
plt.xticks(rotation=90)
plt.legend()

plt.xlabel('Country')
plt.ylabel('Percentage of people who\n earn more or less than 50k');
newAdult = adult
newAdultTest = adultTest
def cutData(data, listOfIndexes):
    base = data.iloc[:,0:0]
    for index, element in enumerate(listOfIndexes):
        if element == 1:
            base = pd.concat([base, data.iloc[:,index:index+1]], sort=False, axis=1)
    return base

def getBestClassifier(featuresList, KRange, trainingData, testingData):
    
    XAdult = cutData(trainingData, featuresList)
    YAdult = trainingData.iloc[:,14:15]

    XAdultTest = cutData(testingData, featuresList)
    YAdultTest = testingData.iloc[:,14:15]
    
    classifiers = []
    for i in range(KRange[0], KRange[1]+1):
        classifiers.append(KNeighborsClassifier(n_neighbors=i))
        
    for index, element in enumerate(classifiers):
        element.fit(XAdult, YAdult)
        
    predictions = []
    for index, element in enumerate(classifiers):
        predictions.append(element.predict(XAdultTest))
        
    accuracies = []
    for element in predictions:
        accuracies.append(accuracy_score(YAdultTest, element))
    
    return max(accuracies), accuracies.index(max(accuracies))+KRange[0], classifiers[accuracies.index(max(accuracies))]
def sortedFeatures(feature, target):
    blankDict = {}
    countryTargetData = {}
    for index, element in enumerate(feature):
        if element not in countryTargetData:
            if target[index] == '<=50K':
                countryTargetData[element] = [1,0]
            else:
                countryTargetData[element] = [0,1]
        else:
            if target[index] == '<=50K':
                countryTargetData[element][0] +=1
            else:
                countryTargetData[element][1] += 1
    return countryTargetData
for i in (1,3,5,6,7,8,9,13):
    replaceDict = sortedFeatures(newAdult.iloc[:,i:i+1].values.transpose()[0].tolist(),newAdult.iloc[:,newAdult.shape[1]-1:newAdult.shape[1]].values.transpose()[0].tolist())
    for key in replaceDict:
        newAdult = newAdult.replace(key, replaceDict[key][1]/(replaceDict[key][1]+replaceDict[key][0]))
        newAdultTest = newAdultTest.replace(key, replaceDict[key][1]/(replaceDict[key][1]+replaceDict[key][0]))
        print('Replacing {0} for {1}'.format(key, replaceDict[key][1]/(replaceDict[key][1]+replaceDict[key][0]) ))
newAdult = newAdult.replace('>50K', 1)
newAdult = newAdult.replace('<=50K',0)
newAdultTest = newAdultTest.replace('>50K.', 1)
newAdultTest = newAdultTest.replace('<=50K.',0)
newAdultTest.head()
acc, k, bestClass = getBestClassifier(([1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]), (10,30), newAdult, newAdultTest)
print('Best classifier: accuracy {0} at K = {1}'.format(acc,k))
clf = ensemble.RandomForestClassifier()
clf.fit(newAdult[['Age','Workclass','Marital Status', 'Occupation', 'Race','Capital Gain','Capital Loss']], newAdult['Target'])
prediction = clf.predict(newAdultTest[['Age','Workclass','Marital Status', 'Occupation', 'Race','Capital Gain','Capital Loss']])
acc = accuracy_score(newAdultTest['Target'],prediction )
print(acc)
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim = 7, activation = 'relu'))
    model.add(Dense(7, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
clf = KerasClassifier(build_fn=create_model, epochs=150, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=16)
clf.fit(newAdult[['Age','Workclass','Marital Status', 'Occupation', 'Race','Capital Gain','Capital Loss']], newAdult['Target'])
prediction = clf.predict(newAdultTest[['Age','Workclass','Marital Status', 'Occupation', 'Race','Capital Gain','Capital Loss']])
acc = accuracy_score(newAdultTest['Target'],prediction )
print(acc)
clf = AdaBoostClassifier(n_estimators=1000)
clf.fit(newAdult[['Age','Workclass','Marital Status', 'Occupation', 'Race','Capital Gain','Capital Loss']], newAdult['Target'])
prediction = clf.predict(newAdultTest[['Age','Workclass','Marital Status', 'Occupation', 'Race','Capital Gain','Capital Loss']])
acc = accuracy_score(newAdultTest['Target'],prediction )
print(acc)
testData = pd.read_csv('../input/test-data-with-targer/test_data.csv',names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?", skiprows = 1)

refData = pd.read_csv('../input/atividade-4-versao-1/train_data.csv',names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?", skiprows = 1).dropna()
for i in (1,3,5,6,7,8,9,13):
    replaceDict = sortedFeatures(refData.iloc[:,i:i+1].values.transpose()[0].tolist(),refData.iloc[:,refData.shape[1]-1:refData.shape[1]].values.transpose()[0].tolist())
    for key in replaceDict:
        refData = refData.replace(key, replaceDict[key][1]/(replaceDict[key][1]+replaceDict[key][0]))
        testData = testData.replace(key, replaceDict[key][1]/(replaceDict[key][1]+replaceDict[key][0]))
        print('Replacing {0} for {1}'.format(key, replaceDict[key][1]/(replaceDict[key][1]+replaceDict[key][0]) ))
testData = testData.fillna(0)
testData = testData.replace('Never-worked',0)
finalPrediction = clf.predict(testData[["Age", "Workclass", "Marital Status",
        "Occupation",  "Race", "Capital Gain", "Capital Loss"]])
import csv

sendFile = open('toSend.csv', mode = 'w')
csvWriter = csv.writer(sendFile)
csvWriter.writerow(['Id', 'income'])
for index, element in enumerate(finalPrediction):
    csvWriter.writerow([index, '>50K' if element == 1 else '<=50K'])
    
sendFile.close()
