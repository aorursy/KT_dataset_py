# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

train.head(5)
test = pd.read_csv("../input/test.csv")

test.head(5)
lenTest = len(test)

lenTrain = len(train)

print('lenTest is %s - lenTrain %s.'%(lenTest,lenTrain))
survived = train[train['Survived']==1]

survived.head(5)
notSurvived = train[train['Survived']==0]

notSurvived.head(5)
print('Survivors are %s - not survived are %s'%(len(survived),len(notSurvived)))
totalMales = len(train[train['Sex']=='male'].index)

totalFemales = len(train[train['Sex']=='female'].index)

print('Total males are: %s and total females are: %s'%(totalMales, totalFemales))
percMales = round((totalMales/len(train.index))*100,2)

percFemales = round((totalFemales/len(train.index))*100,2)

survivedMales = len(survived[survived['Sex']=='male'].index)

survivedFemales = len(survived.index)-survivedMales

percSurvivedMales = round((survivedMales/len(survived.index))*100,2)

percSurvivedFemales = round((survivedFemales/len(survived.index))*100,2)

print('%% passengers males are %s, %% passengers females are %s. \nSurvivors %% male are %s, female are %s'%(percMales, percFemales,percSurvivedMales, percSurvivedFemales ))
probSurvMale = round(survivedMales/totalMales,2)

probSurvFemale = round(survivedFemales/totalFemales,2)

print('The probability to be a survivor is %s for a male, %s for a female'%(probSurvMale,probSurvFemale))
passGroupedByAge = pd.DataFrame({'Count':train.groupby('Age').size()}).reset_index()

survGroupedByAge = pd.DataFrame({'Count':survived.groupby('Age').size()}).reset_index()

print('The passengers of Titanic can be divided in %s groups by age'%len(passGroupedByAge))

print('The survivors of Titanic can be divided in %s groups by age'%len(survGroupedByAge))
survGroupedByAgeRed = survGroupedByAge[['Age','Count']]

survGroupedByAgeRed.set_index('Age')

survGroupedByAgeRed.columns = ['Age','NSurv']



passGroupedByAgeRed = passGroupedByAge[['Age','Count']]

passGroupedByAgeRed.set_index('Age')

passGroupedByAgeRed.columns = ['Age','NPass']



s1 = pd.merge(passGroupedByAgeRed, survGroupedByAgeRed, how='left', on='Age')

s1[['NPass','NSurv']].plot(kind='bar', figsize=(30,20), grid=True)
maxNumAgeGroup = passGroupedByAge['Count'].max()

minNumAgeGroup = passGroupedByAge['Count'].min()

maxAgeGroup = passGroupedByAge['Age'].max()

minAgeGroup = passGroupedByAge['Age'].min()



maxNumAgeSurGroup = survGroupedByAge['Count'].max()

minNumAgeSurGroup = survGroupedByAge['Count'].min()

maxAgeSurGroup = survGroupedByAge['Age'].max()

minAgeSurGroup = survGroupedByAge['Age'].min()



print('Max num of passenger in a age group is %s, min is %s'%(maxNumAgeGroup,minNumAgeGroup))

print('Max age of passenger is %s, min is %s'%(maxAgeGroup,minAgeGroup))

print('Max num of survivors in a age group is %s, min is %s'%(maxNumAgeSurGroup,minNumAgeSurGroup))

print('Max age of survivors is %s, min is %s'%(maxAgeSurGroup,minAgeSurGroup))
print('The survivors of Titanic can be divided in %s groups by age'%len(survGroupedByAge))
numPassAge0_1 = len(train[(train['Age'] > 0) & (train['Age'] < 1)])

numPassAge1_18 = len(train[(train['Age'] >= 1) & (train['Age'] < 18)])

numPassAge18_80 = len(train[(train['Age'] >= 18) & (train['Age'] <= 80)])

numPassUnkAge = len(train[train['Age'].isnull()])



numPassSurAge0_1 = len(survived[(survived['Age'] > 0) & (survived['Age'] < 1)])

numPassSurAge1_18 = len(survived[(survived['Age'] >= 1) & (survived['Age'] < 18)])

numPassSurAge18_80 = len(survived[(survived['Age'] >= 18) & (survived['Age'] <= 80)])

numPassSurUnkAge = len(survived[survived['Age'].isnull()])



print('The prob to survive for a passenger 0-1 aged is %s'%round(numPassSurAge0_1/numPassAge0_1,2))

print('The prob to survive for a passenger 1-18 aged is %s'%round(numPassSurAge1_18/numPassAge1_18,2))

print('The prob to survive for a passenger 18-80 aged is %s'%round(numPassSurAge18_80/numPassAge18_80,2))

print('The prob to survive for a passenger unknown aged is %s'%round(numPassSurUnkAge/numPassUnkAge,2))

numPassAge0_1F = len(train[(train['Age'] > 0) & (train['Age'] < 1) & (train['Sex']=='female')])

numPassAge1_18F = len(train[(train['Age'] >= 1) & (train['Age'] < 18) & (train['Sex']=='female')])

numPassAge18_80F = len(train[(train['Age'] >= 18) & (train['Age'] <= 80) & (train['Sex']=='female')])

numPassUnkAgeF = len(train[(train['Age'].isnull()) & (train['Sex']=='female')])



numPassSurAge0_1F = len(survived[(survived['Age'] > 0) & (survived['Age'] < 1) & (survived['Sex']=='female')])

numPassSurAge1_18F = len(survived[(survived['Age'] >= 1) & (survived['Age'] < 18) & (survived['Sex']=='female')])

numPassSurAge18_80F = len(survived[(survived['Age'] >= 18) & (survived['Age'] <= 80) & (survived['Sex']=='female')])

numPassSurUnkAgeF = len(survived[(survived['Age'].isnull()) & (survived['Sex']=='female')])



print('The prob to survive for a Female 0-1 aged is %s'%round(numPassSurAge0_1F/numPassAge0_1F,2))

print('The prob to survive for a Female 1-18 aged is %s'%round(numPassSurAge1_18F/numPassAge1_18F,2))

print('The prob to survive for a Female 18-80 aged is %s'%round(numPassSurAge18_80F/numPassAge18_80F,2))

print('The prob to survive for a Female unknown aged is %s'%round(numPassSurUnkAgeF/numPassUnkAgeF,2))

numPassAge0_1M = len(train[(train['Age'] > 0) & (train['Age'] < 1) & (train['Sex']=='male')])

numPassAge1_18M = len(train[(train['Age'] >= 1) & (train['Age'] < 18) & (train['Sex']=='male')])

numPassAge18_80M = len(train[(train['Age'] >= 18) & (train['Age'] <= 80) & (train['Sex']=='male')])

numPassUnkAgeM = len(train[(train['Age'].isnull()) & (train['Sex']=='male')])



numPassSurAge0_1M = len(survived[(survived['Age'] > 0) & (survived['Age'] < 1) & (survived['Sex']=='male')])

numPassSurAge1_18M = len(survived[(survived['Age'] >= 1) & (survived['Age'] < 18) & (survived['Sex']=='male')])

numPassSurAge18_80M = len(survived[(survived['Age'] >= 18) & (survived['Age'] <= 80) & (survived['Sex']=='male')])

numPassSurUnkAgeM = len(survived[(survived['Age'].isnull()) & (survived['Sex']=='male')])



print('The prob to survive for a male 0-1 aged is %s'%round(numPassSurAge0_1M/numPassAge0_1M,2))

print('The prob to survive for a male 1-18 aged is %s'%round(numPassSurAge1_18M/numPassAge1_18M,2))

print('The prob to survive for a male 18-80 aged is %s'%round(numPassSurAge18_80M/numPassAge18_80M,2))

print('The prob to survive for a male unknown aged is %s'%round(numPassSurUnkAgeM/numPassUnkAgeM,2))
numPass1stClass = len(train[(train['Pclass'] == 1)])

numPass2stClass = len(train[(train['Pclass'] == 2)])

numPass3stClass = len(train[(train['Pclass'] == 3)])



numPass1stSurClass = len(survived[(survived['Pclass'] == 1)])

numPass2stSurClass = len(survived[(survived['Pclass'] == 2)])

numPass3stSurClass = len(survived[(survived['Pclass'] == 3)])



print('The prob to survive for a pass 1st class is %s'%round(numPass1stSurClass/numPass1stClass,2))

print('The prob to survive for a pass 2st class is %s'%round(numPass2stSurClass/numPass2stClass,2))

print('The prob to survive for a pass 3st class is %s'%round(numPass3stSurClass/numPass3stClass,2))
numPass1STAge0_1F = len(train[(train['Age'] > 0) & (train['Age'] < 1) & (train['Sex']=='female') & (train['Pclass'] == 1)])

numPass1STAge1_18F = len(train[(train['Age'] >= 1) & (train['Age'] < 18) & (train['Sex']=='female') & (train['Pclass'] == 1)])

numPass1STAge18_80F = len(train[(train['Age'] >= 18) & (train['Age'] <= 80) & (train['Sex']=='female') & (train['Pclass'] == 1)])

numPassUnk1STAgeF = len(train[(train['Age'].isnull()) & (train['Sex']=='female') & (train['Pclass'] == 1)])



print('Num passengers female 0-1 year aged in 1st class: %s'%numPass1STAge0_1F)

print('Num passengers female 1-18 year aged in 1st class: %s'%numPass1STAge1_18F)

print('Num passengers female 18-80 year aged in 1st class: %s'%numPass1STAge18_80F)

print('Num passengers female unknown age in 1st class: %s'%numPassUnk1STAgeF)
numPassSur1STAge1_18F = len(survived[(survived['Age'] >= 1) & (survived['Age'] < 18) & (survived['Sex']=='female') & (survived['Pclass'] == 1)])

numPassSur1STAge18_80F = len(survived[(survived['Age'] >= 18) & (survived['Age'] <= 80) & (survived['Sex']=='female') & (survived['Pclass'] == 1)])

numPassSurUnk1STAgeF = len(survived[(survived['Age'].isnull()) & (survived['Sex']=='female') & (survived['Pclass'] == 1)])



print('The prob to survive for a 1st class Female 1-18 aged is %s'%round(numPassSur1STAge1_18F/numPass1STAge1_18F,2))

print('The prob to survive for a 1st class Female 18-80 aged is %s'%round(numPassSur1STAge18_80F/numPass1STAge18_80F,2))

print('The prob to survive for a 1st class Female unknown aged is %s'%round(numPassSurUnk1STAgeF/numPassUnk1STAgeF,2))

numPass2STAge0_1F = len(train[(train['Age'] > 0) & (train['Age'] < 1) & (train['Sex']=='female') & (train['Pclass'] == 2)])

numPass2STAge1_18F = len(train[(train['Age'] >= 1) & (train['Age'] < 18) & (train['Sex']=='female') & (train['Pclass'] == 2)])

numPass2STAge18_80F = len(train[(train['Age'] >= 18) & (train['Age'] <= 80) & (train['Sex']=='female') & (train['Pclass'] == 2)])

numPassUnk2STAgeF = len(train[(train['Age'].isnull()) & (train['Sex']=='female') & (train['Pclass'] == 2)])



print('Num passengers female 0-1 year aged in 2st class: %s'%numPass2STAge0_1F)

print('Num passengers female 1-18 year aged in 2st class: %s'%numPass2STAge1_18F)

print('Num passengers female 18-80 year aged in 2st class: %s'%numPass2STAge18_80F)

print('Num passengers female unknown age in 2st class: %s'%numPassUnk2STAgeF)
numPassSur2STAge1_18F = len(survived[(survived['Age'] >= 1) & (survived['Age'] < 18) & (survived['Sex']=='female') & (survived['Pclass'] == 2)])

numPassSur2STAge18_80F = len(survived[(survived['Age'] >= 18) & (survived['Age'] <= 80) & (survived['Sex']=='female') & (survived['Pclass'] == 2)])

numPassSurUnk2STAgeF = len(survived[(survived['Age'].isnull()) & (survived['Sex']=='female') & (survived['Pclass'] == 2)])



print('The prob to survive for a 2st class Female 1-18 aged is %s'%round(numPassSur2STAge1_18F/numPass2STAge1_18F,2))

print('The prob to survive for a 2st class Female 18-80 aged is %s'%round(numPassSur2STAge18_80F/numPass2STAge18_80F,2))

print('The prob to survive for a 2st class Female unknown aged is %s'%round(numPassSurUnk2STAgeF/numPassUnk2STAgeF,2))

numPass3STAge0_1F = len(train[(train['Age'] > 0) & (train['Age'] < 1) & (train['Sex']=='female') & (train['Pclass'] == 3)])

numPass3STAge1_18F = len(train[(train['Age'] >= 1) & (train['Age'] < 18) & (train['Sex']=='female') & (train['Pclass'] == 3)])

numPass3STAge18_80F = len(train[(train['Age'] >= 18) & (train['Age'] <= 80) & (train['Sex']=='female') & (train['Pclass'] == 3)])

numPassUnk3STAgeF = len(train[(train['Age'].isnull()) & (train['Sex']=='female') & (train['Pclass'] == 3)])



print('Num passengers female 0-1 year aged in 3st class: %s'%numPass3STAge0_1F)

print('Num passengers female 1-18 year aged in 3st class: %s'%numPass3STAge1_18F)

print('Num passengers female 18-80 year aged in 3st class: %s'%numPass3STAge18_80F)

print('Num passengers female unknown age in 3st class: %s'%numPassUnk3STAgeF)
numPassSur3STAge0_1F = len(survived[(survived['Age'] > 0) & (survived['Age'] < 1) & (survived['Sex']=='female') & (survived['Pclass'] == 3)])

numPassSur3STAge1_18F = len(survived[(survived['Age'] >= 1) & (survived['Age'] < 18) & (survived['Sex']=='female') & (survived['Pclass'] == 3)])

numPassSur3STAge18_80F = len(survived[(survived['Age'] >= 18) & (survived['Age'] <= 80) & (survived['Sex']=='female') & (survived['Pclass'] == 3)])

numPassSurUnk3STAgeF = len(survived[(survived['Age'].isnull()) & (survived['Sex']=='female') & (survived['Pclass'] == 3)])



print('The prob to survive for a 3st class Female 0-1 aged is %s'%round(numPassSur3STAge0_1F/numPass3STAge0_1F,2))

print('The prob to survive for a 3st class Female 1-18 aged is %s'%round(numPassSur3STAge1_18F/numPass3STAge1_18F,2))

print('The prob to survive for a 3st class Female 18-80 aged is %s'%round(numPassSur3STAge18_80F/numPass3STAge18_80F,2))

print('The prob to survive for a 3st class Female unknown aged is %s'%round(numPassSurUnk3STAgeF/numPassUnk3STAgeF,2))
