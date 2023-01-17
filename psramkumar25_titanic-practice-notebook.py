# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv # CSV file operations



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

trainFile = open('../input/train.csv','r')

train = csv.reader(trainFile)

header = str(next(train))

data = []

for row in train:

    data.append(row)    

data = np.array(data)

trainFile.close()
NumPassengers = data[:,1].size

NumSurvivors = sum(data[:,1].astype('int'))

PctSurvived = (NumSurvivors/NumPassengers)*100

print('Only {} people survived out of the total {} passengers, which is a shocking {:.4f}%'.format(NumSurvivors,NumPassengers,PctSurvived))

femaleSel = data[:,4] == 'female'

maleSel = data[:,4] == 'male'

MaleData = data[maleSel,1].astype('int')

#print(MaleData,MaleData.size)

NumMale = MaleData.size

MaleSur = MaleData.sum()

NumFemale = data[femaleSel,0].size

FemaleSur = data[femaleSel,1].astype('int').sum()

print(NumMale,MaleSur,NumFemale,FemaleSur)

testFile = open('../input/test.csv','r')

test = csv.reader(testFile)

testHeader = str(next(test))

testData = []

outputFile = open('../input/genderbasedmodel.csv','w')

output = csv.writer(outputFile)

output.writerow(['PassengerId','Survived'])

for row in test:

    testData.append(row)

    if row[3] == 'male':

        output.writerow(row[0],'0')

    elif row[3] == 'female':

        output.writerow(row[0],'1')

testFile.close()

outputFile.close()


z = np.array([1,2,3,4,5,6,7,8,9],'int')

even = z%2 == 0

odd  = z%2 != 0

print(z[even],z[odd])

z = z.reshape((3,3))

e = z[:,0]%2 == 0

o = z[:,0]%2 != 0

print(z[e,2])

print(z[o,2])
