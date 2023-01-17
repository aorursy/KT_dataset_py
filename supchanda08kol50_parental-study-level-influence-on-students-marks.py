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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
df= pd.read_csv('../input/StudentsPerformance.csv')
SeriesOfMathsMean=pd.Series()
SeriesOfParentalDegree=pd.Series()
for key,value in df.groupby(['parental level of education']):
    SeriesOfParentalDegree = SeriesOfParentalDegree.append(pd.Series(key),ignore_index=True)
    SeriesOfMathsMean= SeriesOfMathsMean.append(pd.Series((value['math score']).mean()),ignore_index=True)
MeanOfMathsWithParentalEducation = pd.concat([pd.DataFrame(SeriesOfParentalDegree),pd.DataFrame(SeriesOfMathsMean)],axis=1)
MeanOfMathsWithParentalEducation.columns=['ParentalLevelOfEducation','Mean marks in Maths']
MeanOfMathsWithParentalEducation=MeanOfMathsWithParentalEducation.sort_values(by=['Mean marks in Maths'],ascending=False,inplace=False).reset_index(drop=True,inplace=False)
display(MeanOfMathsWithParentalEducation)
plt.figure(figsize=(16,6))
sns.barplot(x=MeanOfMathsWithParentalEducation['ParentalLevelOfEducation'],y=MeanOfMathsWithParentalEducation['Mean marks in Maths'],linewidth=2.5,palette='ocean',errcolor=".9", edgecolor=".07" )
plt.xlabel('Parental Level Of Education',fontsize=18)
plt.ylabel('Mean Marks in Maths of their Children',fontsize=18)
plt.title('Parental Education vs their Children\'s marks in Maths')
plt.show()
SeriesOfParentalDegree=pd.Series()
SeriesOfReadingsMean=pd.Series()
for key,value in df.groupby(['parental level of education']):
    SeriesOfParentalDegree = SeriesOfParentalDegree.append(pd.Series(key),ignore_index=True)
    SeriesOfReadingsMean= SeriesOfReadingsMean.append(pd.Series((value['reading score']).mean()),ignore_index=True)
MeanOfReadingsWithParentalEducation = pd.concat([pd.DataFrame(SeriesOfParentalDegree),pd.DataFrame(SeriesOfReadingsMean)],axis=1)
MeanOfReadingsWithParentalEducation.columns=['ParentalLevelOfEducation','Mean marks in Reading']
MeanOfReadingsWithParentalEducation=MeanOfReadingsWithParentalEducation.sort_values(by=['Mean marks in Reading'],ascending=False,inplace=False).reset_index(drop=True,inplace=False)
display(MeanOfReadingsWithParentalEducation)
plt.figure(figsize=(16,6))
sns.barplot(x=MeanOfReadingsWithParentalEducation['ParentalLevelOfEducation'],y=MeanOfReadingsWithParentalEducation['Mean marks in Reading'],linewidth=2.5,palette='ocean_r',errcolor=".9", edgecolor=".07" )
plt.xlabel('Parental Level Of Education',fontsize=18)
plt.ylabel('Mean Marks in Reading of their Children',fontsize=18)
plt.title('Parental Education vs their Children marks in Reading')
plt.show()

SeriesOfParentalDegree=pd.Series()
SeriesOfReadingsMean=pd.Series()
for key,value in df.groupby(['parental level of education']):
    SeriesOfParentalDegree = SeriesOfParentalDegree.append(pd.Series(key),ignore_index=True)
    SeriesOfReadingsMean= SeriesOfReadingsMean.append(pd.Series((value['writing score']).mean()),ignore_index=True)
MeanOfReadingsWithParentalEducation = pd.concat([pd.DataFrame(SeriesOfParentalDegree),pd.DataFrame(SeriesOfReadingsMean)],axis=1)
MeanOfReadingsWithParentalEducation.columns=['ParentalLevelOfEducation','Mean marks in writing']
MeanOfReadingsWithParentalEducation=MeanOfReadingsWithParentalEducation.sort_values(by=['Mean marks in writing'],ascending=False,inplace=False).reset_index(drop=True,inplace=False)
display(MeanOfReadingsWithParentalEducation)
plt.figure(figsize=(16,6))
sns.barplot(x=MeanOfReadingsWithParentalEducation['ParentalLevelOfEducation'],y=MeanOfReadingsWithParentalEducation['Mean marks in writing'],linewidth=2.5,palette='plasma',errcolor=".9", edgecolor=".07" )
plt.xlabel('Parental Level Of Education',fontsize=18)
plt.ylabel('Mean Marks in Reading of their Children',fontsize=18)
plt.title('Parental Education vs their Children marks in writing')
plt.show()
testPrepKey=pd.Series()
testPrepValue=pd.Series()
for key,value in df.groupby(['test preparation course']):
    testPrepKey = testPrepKey.append(pd.Series(key),ignore_index=True)
    testPrepValue= testPrepValue.append(pd.Series((value['math score']).mean()),ignore_index=True)
k = pd.concat([pd.DataFrame(testPrepKey),pd.DataFrame(testPrepValue)],axis=1)
k.columns=['test preparation course','Mean marks in math']
k=k.sort_values(by=['Mean marks in math'],ascending=False,inplace=False).reset_index(drop=True,inplace=False)
display(k)
plt.figure(figsize=(6,3))
sns.barplot(x=k['test preparation course'],y=k['Mean marks in math'],linewidth=2.5,palette='plasma',errcolor=".9", edgecolor=".07" )
plt.xlabel('Test preparation course',fontsize=18)
plt.ylabel('Mean Marks in math of their Children',fontsize=10)
plt.title('test preparation course vs their Children marks in math')
plt.show()

testPrepKey=pd.Series()
testPrepValue=pd.Series()
for key,value in df.groupby(['test preparation course']):
    testPrepKey = testPrepKey.append(pd.Series(key),ignore_index=True)
    testPrepValue= testPrepValue.append(pd.Series((value['reading score']).mean()),ignore_index=True)
k = pd.concat([pd.DataFrame(testPrepKey),pd.DataFrame(testPrepValue)],axis=1)
k.columns=['test preparation course','Mean marks in reading']
k=k.sort_values(by=['Mean marks in reading'],ascending=False,inplace=False).reset_index(drop=True,inplace=False)
display(k)
plt.figure(figsize=(6,3))
sns.barplot(x=k['test preparation course'],y=k['Mean marks in reading'],linewidth=2.5,palette='plasma',errcolor=".9", edgecolor=".07" )
plt.xlabel('Test preparation course',fontsize=18)
plt.ylabel('Mean Marks in reading of their Children',fontsize=10)
plt.title('test preparation course vs their Children marks in reading')
plt.show()

testPrepKey=pd.Series()
testPrepValue=pd.Series()
for key,value in df.groupby(['test preparation course']):
    testPrepKey = testPrepKey.append(pd.Series(key),ignore_index=True)
    testPrepValue= testPrepValue.append(pd.Series((value['writing score']).mean()),ignore_index=True)
k = pd.concat([pd.DataFrame(testPrepKey),pd.DataFrame(testPrepValue)],axis=1)
k.columns=['test preparation course','Mean marks in writing']
k=k.sort_values(by=['Mean marks in writing'],ascending=False,inplace=False).reset_index(drop=True,inplace=False)
display(k)
plt.figure(figsize=(6,3))
sns.barplot(x=k['test preparation course'],y=k['Mean marks in writing'],linewidth=2.5,palette='plasma',errcolor=".9", edgecolor=".07" )
plt.xlabel('Test preparation course',fontsize=18)
plt.ylabel('Mean Marks in writing of their Children',fontsize=10)
plt.title('test preparation course vs their Children marks in writing')
plt.show()
sex= pd.Series()
pdOverAll=pd.DataFrame()
for key,value in df.groupby(['gender']):
    sex = sex.append(pd.Series(key))
    sex = sex.append(pd.Series(key).copy())
    pdOverAll = (pd.concat([pdOverAll,pd.DataFrame(value['test preparation course'].value_counts())],axis=0))
pdOverAll = pdOverAll.reset_index(drop=False,inplace=False)
sex= pd.DataFrame(sex).rename(columns={0:'Sex'}).reset_index(drop=True)
pdOverAll= pd.concat([sex,pdOverAll],axis=1)
pdOverAll.rename(columns={'index':'Test Preparation Course','test preparation course':'Test Preparation Count'},inplace=True)
display(pdOverAll)

plt.figure(figsize=(16,6))
pdOverAll = pdOverAll.pivot('Sex','Test Preparation Course','Test Preparation Count')
sns.heatmap(pdOverAll,annot=True,cbar=True,fmt='.0f',cmap='gist_heat',linewidth= 4.0,linecolor='grey')
plt.show()
plt.figure(figsize=(14,8))
sns.boxplot(x= df['parental level of education'],y=df['math score'])
sns.swarmplot(x= df['parental level of education'],y=df['math score'],edgecolor='violet',linewidth = 1.0)
plt.show()
plt.figure(figsize=(14,8))
sns.boxplot(x= df['parental level of education'],y=df['reading score'])
sns.swarmplot(x= df['parental level of education'],y=df['reading score'],edgecolor='black',linewidth = 1.0)
plt.show()
plt.figure(figsize=(14,8))
sns.boxplot(x= df['parental level of education'],y=df['writing score'])
sns.swarmplot(x= df['parental level of education'],y=df['writing score'],edgecolor='yellow',linewidth = 1.0)
plt.show()
