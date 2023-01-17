import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import linear_model
import csv
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Map containing numerical equivalent of different titles and countries.
#Master --> Bachelors
#Don, Rev, Dr --> 2
#Sir, Lady, Countess --> Royalty(3)
# Col, Capt, Major --> Military(4)
#Miss, Ms, Mrs, MMe, Mlle-- 5
#Mr -- 6
titleRespect = {'Mr' : 6,  'Mrs': 5, 'Miss': 5,  'Master': 1,  'Don' : 2,  'Rev': 2,  'Dr': 2,  'Mme': 5,'Ms': 5, 'Major' : 4,
 'Lady': 3, 'Sir': 3,  'Mlle': 5,  'Col': 4, 'Capt': 4, 'the Countess': 3,  'Jonkheer': 3}

# Englih common -- 1
# Don -- Spanish title(2)
# Mme, Mlle -- French(3)
# Jonkheer -- Dutch(4)
titleNational = {'Mr' : 1,  'Mrs': 1, 'Miss': 1,  'Master': 1,  'Don' : 2,  'Rev': 1,  'Dr': 1,  'Mme': 3 ,'Ms': 1, 'Major' : 1,
 'Lady': 1, 'Sir':1,  'Mlle': 3,  'Col': 1, 'Capt': 1, 'the Countess': 1,  'Jonkheer': 4}


#Splits the name to last name, title and first names.
def splitName(x):
    lnFn  = re.split(',', x)
    lastName = lnFn[0]
    firstName = lnFn[1]

    titleFn = re.split('\.', firstName)
    title = titleFn[0]
    firstName = titleFn[1]
    return lastName, title, firstName
    
#refers to the maps defined above and returns respective integer equivalents.
def titleMapping(x):
    return titleRespect.get(x,1), titleNational.get(x, 1)

#splits a continous value into buckets specified by numberofBuckets parameter.
def discretizeColumn(columnName, numberOfBuckets, frame):
    maxValue = frame[columnName].max()
    minValue = frame[columnName].min()
    bucketSize = (maxValue - minValue)/numberOfBuckets
    frame[columnName] =  np.ceil((frame[columnName] / bucketSize))
    frame.loc[frame[columnName] > numberOfBuckets - 1, columnName] = numberOfBuckets - 1

#displays integer value of bars in the charts.
def autoLabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05* height, '%d' %int(height), ha='center', va='bottom')

#Function to draw a bar chart of the columnName specified vs the class variable.
#The labels displayed are passed as a parameter.
def drawBarChart(columnName, classVariable, legendLabels, frame):
    numberClasses = frame[columnName].unique()
    survivedCount = []
    notSurvivedCount = []

    for pclass in numberClasses:
        survivedCount.append(len(frame[[ a and b for(a, b) in zip(frame[columnName] == pclass, frame[classVariable] == 1 )]]))
        notSurvivedCount.append(len(frame[[ a and b for(a, b) in zip(frame[columnName] == pclass, frame[classVariable] == 0 )]]))

    width = 0.2
    fig, ax = plt.subplots()
    rects1 = ax.bar(numberClasses, survivedCount, width, color='b')
    rects2 = ax.bar(numberClasses + width, notSurvivedCount, width, color='r')
    ax.set_ylabel('Count')
    ax.set_xticks(numberClasses + width)
    ax.set_xticklabels(tuple(numberClasses))
    ax.set_xlabel(columnName)
    ax.legend((rects1[0], rects2[0]), legendLabels)

    autoLabel(rects1, ax)
    autoLabel(rects2, ax)

    plt.show()

#Single function that calls drawBarChart on various columns with respect to the class variable.
def visualizeData(frame):
    #continous values - Age, Fare: Discretize and plot
    drawBarChart('Age', 'Survived',('Survived', 'Not Survived'), frame)
    drawBarChart('Fare', 'Survived',('Survived', 'Not Survived'), frame)
    drawBarChart('FamilySize', 'Survived', ('Survived', 'Not Survived'), frame)
    #discrete values - Pclass, Sex, Embarked
    drawBarChart('Pclass', 'Survived', ('Survived', 'Not Survived'), frame)
    drawBarChart('Sex', 'Survived', ('Survived', 'Not Survived'),frame)
    drawBarChart('Embarked', 'Survived', ('Survived', 'Not Survived'), frame)
    drawBarChart('Respected', 'Survived', ('Survived', 'Not Survived'), frame)
    drawBarChart('Nation', 'Survived', ('Survived', 'Not Survived'), frame)
    drawBarChart('TicketCount', 'Survived', ('Survived', 'Not Survived'), frame)

#Fills missing values with acceptable entries. 
def fillMissingValues(frame):
    #fill missing values Fare is dependent on the class of the ticket.
    frame['Fare'].fillna(frame['Pclass']*10, inplace=True)
    frame['Embarked'].fillna('S', inplace=True)
    frame['Parch'].fillna(0,inplace=True)
    frame['SibSp'].fillna(0, inplace=True)
    frame['FamilySize'] = frame['Parch'] + frame['SibSp']
    
#Replaces string with equivalent integers. This is done because some algorithms doesn't work on string features.
def mapDiscreteColumns(frame):
    #Replace all strings with representative numbers.
    genderMapping = {'male': 0, 'female': 1}
    frame.replace({'Sex' : genderMapping}, inplace=True)
    embarkedMapping = {'S': 0, 'Q': 1, 'C':2}
    frame.replace({'Embarked' : embarkedMapping}, inplace=True)

#Single function which calls discretizeColumn on different columns.
def discretizeColumns(frame):
    discretizeColumn('Age', 5 , frame)
    discretizeColumn('Fare', 4, frame)
    discretizeColumn('FamilySize', 3, frame)

#Handle missing values of column age.
#Age is dependent on the fare attribute. So, we use linear regression to predict age of missing values based on
# the fares.
def handleAgeMissingValues(frame):
    Xtest = frame[frame['Age'].isnull()][['Fare']]
    X = frame[frame['Age'].notnull()][['Fare']]
    y = frame[frame['Age'].notnull()][['Age']]
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    ytest = lr.predict(Xtest)
    frame.loc[frame['Age'].isnull(), 'Age'] = ytest

#read the training data.    
complete_train_data = pd.read_csv('../input/train.csv', header= 0)
fillMissingValues(complete_train_data)
mapDiscreteColumns(complete_train_data)

#handle Age missing values differently
handleAgeMissingValues(complete_train_data)
discretizeColumns(complete_train_data)

complete_train_data['LastName'],complete_train_data['Title'],complete_train_data['FirstName'] = zip(*complete_train_data['Name'].map(splitName))
complete_train_data['Title'] = complete_train_data['Title'].str.strip()
complete_train_data['Respected'], complete_train_data['Nation'] = zip(*complete_train_data['Title'].map(titleMapping))
complete_train_data['TicketCount'] = complete_train_data.groupby('Ticket')['Ticket'].transform('count')

visualizeData(complete_train_data)

#Assume 66% of the training set as training data.
#Rest 34% becomes cross validation set.
m = len(complete_train_data)
train_data_size = int(np.ceil(0.66 * m));
cross_val_size = m - train_data_size

train_data = complete_train_data[0:train_data_size]
cross_val_data = complete_train_data[train_data_size + 1:m]

X = train_data[['Sex','Age', 'Fare', 'Embarked', 'Pclass', 'FamilySize', 'Respected']]
y = train_data[['Survived']]

#Instantiate a classifier.
#clf = svm.SVC(verbose = True)
#clf = RandomForestClassifier(random_state = 10, warm_start = True, n_estimators = 26, max_depth = 6, max_features = 'sqrt')
clf = GradientBoostingClassifier()
clf.fit(X, y)

XcrossVal = cross_val_data[['Sex','Age', 'Fare', 'Embarked', 'Pclass', 'FamilySize', 'Respected']]
ycrossVal = cross_val_data[['Survived']]

# run predictions on the cross validation set and calculate accuracy.
predictions = clf.predict(XcrossVal)
predictions = np.reshape(predictions, np.shape(ycrossVal))

results = (predictions == ycrossVal)
trueCount = np.count_nonzero(results)
totalCount = len(results)

accuracy = trueCount * 100.0 / totalCount

#read test data and perform preprocessing.
test_data = pd.read_csv('../input/test.csv', header = 0)
fillMissingValues(test_data)
mapDiscreteColumns(test_data)
handleAgeMissingValues(test_data)
discretizeColumns(test_data)
test_data['LastName'],test_data['Title'],test_data['FirstName'] = zip(*test_data['Name'].map(splitName))
test_data['Title'] = test_data['Title'].str.strip()
test_data['Respected'], test_data['Nation'] = zip(*test_data['Title'].map(titleMapping))
test_data['TicketCount'] = test_data.groupby('Ticket')['Ticket'].transform('count')

XTest = test_data[['Sex','Age', 'Fare', 'Embarked', 'Pclass', 'FamilySize','Respected']]

#run the classifier on the test data.
predictions = clf.predict(XTest)

prediction_file = open('results.csv','w')
writer_obj = csv.writer(prediction_file)
writer_obj.writerow(['PassengerId', 'Survived'])

for i in range(len(predictions)):
    writer_obj.writerow([test_data['PassengerId'][i], predictions[i]])

prediction_file.close()
