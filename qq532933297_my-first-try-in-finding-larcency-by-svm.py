import numpy as np 
import pandas as pd 
import zipfile
from sklearn import svm

drive = '../input/'

# Read data
z = zipfile.ZipFile(drive+'train.csv.zip')
train = pd.read_csv(z.open('train.csv'))
z = zipfile.ZipFile(drive+'test.csv.zip')
test = pd.read_csv(z.open('test.csv'))
maxIdx = train.shape[0]-1
sampler = np.random.permutation(maxIdx)
train = train.take(sampler[:int(0.05*maxIdx)])
maxIdx = test.shape[0]-1
sampler = np.random.permutation(maxIdx)
test = test.take(sampler[:int(0.001*maxIdx)])
test_old = test

# Change string to number
train.Category[train.Category != 'LARCENY/THEFT'] = -1
train.Category[train.Category == 'LARCENY/THEFT'] = 1

daySet = train.DayOfWeek.unique()
i=1
dayInWeek =['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'] 
for day in daySet:
    train.DayOfWeek[train.DayOfWeek == day] = dayInWeek.index(day)
    test.DayOfWeek[test.DayOfWeek == day] = dayInWeek.index(day)
    
districtSet = train.PdDistrict.unique()
i=1
for district in districtSet:
    train.PdDistrict[train.PdDistrict == district] = i
    test.PdDistrict[test.PdDistrict == district] = i
    i = i+1

train['X'] = list([round(val*100.0) for val in train.X.values])
train['Y'] = list([round(val*100.0) for val in train.Y.values])
test['X'] = list([round(val*100.0) for val in test.X.values])
test['Y'] = list([round(val*100.0) for val in test.Y.values])

# Generate the train set and the validation set
trainSet = train[:int(train.shape[0]*0.75)]
validationSet = train[int(train.shape[0]*0.75):]


# 
clf = svm.SVC(kernel='linear', C=1)
clf.fit(np.array(trainSet[['DayOfWeek','PdDistrict','X','Y']].values.tolist()),np.array(trainSet['Category'].values.tolist()))
predictedResult = clf.predict(np.array(validationSet[['DayOfWeek','PdDistrict','X','Y']].values.tolist()))
Table = validationSet[['DayOfWeek','PdDistrict','X','Y','Category']]
Table['Crime'] = predictedResult
print(Table)

Table.to_csv('submit.csv')

