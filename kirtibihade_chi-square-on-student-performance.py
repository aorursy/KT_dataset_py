## Purpose of this notebook is to check in there any dependancy between score and gender

## Between test preparation course and score

## Between score and parental level of education

## Between lunch type and gender(Just to check)



### Link for chi square table https://www.medcalc.org/manual/chi-square-table.php
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as sps
StudentPerformanceData =  pd.read_csv("../input/StudentsPerformanceInExam.csv")
StudentPerformanceData.head()
StudentPerformanceDataFemale = StudentPerformanceData.loc[StudentPerformanceData['gender']=='female',:]
StudentPerformanceDataMale = StudentPerformanceData.loc[StudentPerformanceData['gender']=='male',:]
sns.distplot(StudentPerformanceDataMale['math score'],color='green', label="male")

sns.distplot(StudentPerformanceDataFemale['math score'],color='blue', label="female")

plt.legend()
# Convert numeric math score into categorical with 4 categories Low ,Medium ,High ,Excellent

StudentPerformanceData['math score categories'] = pd.cut(StudentPerformanceData['math score'] ,4,labels = ['Low','Medium','High','Excellent'])
Zippedlist = list(zip(StudentPerformanceData['gender'],StudentPerformanceData['math score categories'])) 

GenderMathScore = pd.DataFrame(Zippedlist,columns=['gender','math score categories'])

GenderMathScorePivot = GenderMathScore.reset_index().groupby(['gender','math score categories']).count().reset_index()

GenderMathScorePivot

GenderMathScoreContigencyTable = GenderMathScorePivot.pivot(index='gender', columns='math score categories', values='index')

GenderMathScoreContigencyTable

GenderMathScoreContigencyTable.fillna(0,inplace = True)
GenderMathScoreContigencyTable
chi2,p,dof,expected =sps.chi2_contingency(GenderMathScoreContigencyTable, correction=False)

chi2,p,dof,expected
##########################################################
sns.distplot(StudentPerformanceDataMale['reading score'],color='green', label="male")

sns.distplot(StudentPerformanceDataFemale['reading score'],color='blue', label="female")

plt.legend()
StudentPerformanceData['reading score categories'] = pd.cut(StudentPerformanceData['reading score'] ,4,labels = ['Low','Medium','High','Excellent'])
Zippedlist = list(zip(StudentPerformanceData['gender'],StudentPerformanceData['reading score categories'])) 

GenderReadingScore = pd.DataFrame(Zippedlist,columns=['gender','reading score categories'])

GenderReadingScorePivot = GenderReadingScore.reset_index().groupby(['gender','reading score categories']).count().reset_index()

GenderReadingScorePivot

GenderReadingScoreContigencyTable = GenderReadingScorePivot.pivot(index='gender', columns='reading score categories', values='index')

GenderReadingScoreContigencyTable

GenderReadingScoreContigencyTable.fillna(0,inplace = True)
GenderReadingScoreContigencyTable
chi2,p,dof,expected =sps.chi2_contingency(GenderReadingScoreContigencyTable, correction=False)

chi2,p,dof,expected
###########################################################
sns.distplot(StudentPerformanceDataMale['writing score'],color='green', label="male")

sns.distplot(StudentPerformanceDataFemale['writing score'],color='blue', label="female")

plt.legend()
StudentPerformanceData['writing score categories'] = pd.cut(StudentPerformanceData['writing score'] ,4,labels = ['Low','Medium','High','Excellent'])
Zippedlist = list(zip(StudentPerformanceData['gender'],StudentPerformanceData['writing score categories'])) 

GenderWritingScore = pd.DataFrame(Zippedlist,columns=['gender','writing score categories'])

GenderWritingScorePivot = GenderWritingScore.reset_index().groupby(['gender','writing score categories']).count().reset_index()

GenderWritingScorePivot

GenderWritingScoreContigencyTable = GenderWritingScorePivot.pivot(index='gender', columns='writing score categories', values='index')

GenderWritingScoreContigencyTable

GenderWritingScoreContigencyTable.fillna(0,inplace = True)
GenderWritingScoreContigencyTable
chi2,p,dof,expected = sps.chi2_contingency(GenderWritingScoreContigencyTable, correction=False)

chi2,p,dof,expected 
####################################################################
Zippedlist = list(zip(StudentPerformanceData['math score categories'] ,StudentPerformanceData['test preparation course'])) 

TestPrepMathScore = pd.DataFrame(Zippedlist,columns=['math score categories','test preparation course'])

TestPrepMathScorePivot = TestPrepMathScore.reset_index().groupby(['math score categories','test preparation course']).count().reset_index()

TestPrepMathScorePivot

TestPrepMathScoreContigencyTable = TestPrepMathScorePivot.pivot(index='math score categories', columns='test preparation course', values='index')

TestPrepMathScoreContigencyTable 

TestPrepMathScoreContigencyTable .fillna(0,inplace = True)
TestPrepMathScoreContigencyTable
chi2,p,dof,expected = sps.chi2_contingency(TestPrepMathScoreContigencyTable, correction=False)

chi2,p,dof,expected 
###################### You can check for reading and writing score and test preparation 
Zippedlist = list(zip(StudentPerformanceData['parental level of education'],StudentPerformanceData['math score categories'])) 

ParentEduMathScore = pd.DataFrame(Zippedlist,columns=['parental level of education','math score categories'])

ParentEduMathScorePivot = ParentEduMathScore.reset_index().groupby(['parental level of education','math score categories']).count().reset_index()

ParentEduMathScorePivot 

ParentEduMathScoreContigencyTable =ParentEduMathScorePivot .pivot(index='parental level of education', columns='math score categories', values='index')

ParentEduMathScoreContigencyTable

ParentEduMathScoreContigencyTable.fillna(0,inplace = True)
ParentEduMathScoreContigencyTable
chi2,p,dof,expected = sps.chi2_contingency(ParentEduMathScoreContigencyTable, correction=False)

chi2,p,dof,expected 
##############################################################################
Zippedlist = list(zip(StudentPerformanceData['parental level of education'],StudentPerformanceData['writing score categories'])) 

ParentEduWritingScoreCat = pd.DataFrame(Zippedlist,columns=['parental level of education','writing score categories'])

ParentEduWritingScoreCategoriesPivot = ParentEduWritingScoreCat.reset_index().groupby(['parental level of education','writing score categories']).count().reset_index()

ParentEduWritingScoreCategoriesPivot

ContigencyTableParentEduWritingScore = ParentEduWritingScoreCategoriesPivot.pivot(index='parental level of education', columns='writing score categories', values='index')

ContigencyTableParentEduWritingScore

ContigencyTableParentEduWritingScore.fillna(0,inplace = True)
ContigencyTableParentEduWritingScore
chi2,p,dof,expected = sps.chi2_contingency(ContigencyTableParentEduWritingScore, correction=False)

chi2,p,dof,expected 
####################################################################
Zippedlist = list(zip(StudentPerformanceData['parental level of education'],StudentPerformanceData['reading score categories'])) 

ParentEduReadingScoreCat = pd.DataFrame(Zippedlist,columns=['parental level of education','reading score categories'])

ParentEduReadingScoreCategoriesPivot = ParentEduReadingScoreCat.reset_index().groupby(['parental level of education','reading score categories']).count().reset_index()

ParentEduReadingScoreCategoriesPivot

ContigencyTableParentEduReadingScore = ParentEduReadingScoreCategoriesPivot.pivot(index='parental level of education', columns='reading score categories', values='index')

ContigencyTableParentEduReadingScore

ContigencyTableParentEduReadingScore.fillna(0,inplace = True)
ContigencyTableParentEduReadingScore
chi2,p,dof,expected = sps.chi2_contingency(ContigencyTableParentEduReadingScore, correction=False)

chi2,p,dof,expected 
###########################################################################
Zippedlist = list(zip(StudentPerformanceData['gender'],StudentPerformanceData['lunch'])) 

GenderLunch = pd.DataFrame(Zippedlist,columns=['gender','lunch'])

GenderLunchPivot = GenderLunch .reset_index().groupby(['gender','lunch']).count().reset_index()

GenderLunchPivot

GenderLunchContigencyTable =GenderLunchPivot.pivot(index='gender', columns='lunch', values='index')

GenderLunchContigencyTable

GenderLunchContigencyTable.fillna(0,inplace = True)
GenderLunchContigencyTable
chi2,p,dof,expected = sps.chi2_contingency(GenderLunchContigencyTable, correction=False)

chi2,p,dof,expected 
#############################################################################