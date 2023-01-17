# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.ticker as mtick



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Set the stype

plt.style.use('fivethirtyeight')
#Import the 'May 2018 National Occupational Employment and Wage Estimates (United States)'

import pandas as pd

wageAll = pd.read_csv("../input/clean-wage1/clean_wage1.csv")

wageAll
#Filter the table to contain only occupation categories

wageMajor=wageAll[wageAll['Level'].str.contains("major")].reset_index(drop=True)

wageMajor['Annual Mean Wage'] = wageMajor['Annual Mean Wage']

wageMajor
#Filter the table to contain detailed occupations

wageDetailed=wageAll[wageAll['Level'].str.contains("detailed")].reset_index(drop=True)

wageDetailed
#import 'Employment by detailed occupation, race, and Hispanic ethnicity'

raceJob = pd.read_csv("../input/530-clean-race-job/clean_race_job.csv")

raceJob
# import 'Detailed Years of School Completed by People 25 Years and Over by Sex, Age Groups, Race and Hispanic Origin: 2018'

import pandas as pd

eduRace = pd.read_csv("../input/education-race-clean/clean_education.csv")
categorySort = raceJob[['White','Black','Asian','Hispanic or Latino','Category']].groupby('Category').sum().reset_index()

categorySort
#I want to define a function to get rid of the 'occupations' in the table, otherwise the labels will be too long



def replaceString(someobject):

    aList = []

    for n in someobject:

        aList.append(n)

    for i in range(len(aList)):

        aList[i] = aList[i].replace(" occupations",'')

    return aList
wageMajor_n = wageMajor #copy the orginal table

wageMajor_n.index = wageMajor['OCC_TITLE'].str.lower().values #Use the 'OCC_TITLE'(occupation title) to be index
#Top 5 Occupation Categories for Whites(Left)

plt.figure(figsize = (25,15))

axWhite = plt.subplot(421)

axWhite.set(xlim=(0,20))

axWhite.xaxis.set_major_formatter(mtick.PercentFormatter()) #Format the xais ticks into percentage

# get the top 5 most popular occupation categories for Whites

white = categorySort[['White','Category']].sort_values(by = 'White', ascending = True).tail(5) 

plt.barh(replaceString(white['Category']),white['White']/categorySort['White'].sum()*100)

plt.title('Top 5 Occupation Categories for Whites(%)',fontSize = 20)



#Occupation-Related Wagea for Whites(Right)

temp_white = []

for i in white['Category']:

    # Search the top 5 most popular occupation categories in the wage table and get the annual mean wage of that specific category

    # Also maintain the rank order from the left

    temp_white.append(wageMajor_n.loc[i.lower() ,'Annual Mean Wage']) 

axWhiteWage = plt.subplot(422)

plt.barh(replaceString(white['Category']),temp_white,color=['Salmon'])

plt.title('Occupation-Related Wages for Whites',fontSize = 20)



#Top 5 Occupation Categories for Blacks(Left)

plt.subplot(423,sharex = axWhite)

black = categorySort[['Black','Category']].sort_values(by = 'Black', ascending = True).tail(5)

plt.barh(replaceString(black['Category']),black['Black']/categorySort['Black'].sum()*100)

plt.title('Top 5 Occupation Categories for Blacks(%)',fontSize = 20)



#Occupation-Related Wage for Blacks(Right)

temp_black = []

for i in black['Category']:

    temp_black.append(wageMajor_n.loc[i.lower() ,'Annual Mean Wage'])

plt.subplot(424,sharex = axWhiteWage)

plt.barh(replaceString(black['Category']),temp_black,color=['Salmon'])

plt.title('Occupation-Related Wages for Blacks',fontSize = 20)



#Top 5 Occupation Categories for Asians(Left)

plt.subplot(425,sharex = axWhite)

asian = categorySort[['Asian','Category']].sort_values(by = 'Asian', ascending = True).tail(5)

plt.barh(replaceString(asian['Category']),asian['Asian']/categorySort['Asian'].sum()*100)

plt.title('Top 5 Occupation Categories for Asians(%)',fontSize = 20)



#Occupation-Related Wage for Asians(Right)

temp_asian = []

for i in asian['Category']:

    temp_asian.append(wageMajor_n.loc[i.lower() ,'Annual Mean Wage'])

plt.subplot(426,sharex = axWhiteWage)

plt.barh(replaceString(asian['Category']),temp_asian,color=['Salmon'])

plt.title('Occupation-Related Wages for Asians',fontSize = 20)



#Top 5 Occupation Categories for Hispanic or Latino(Left)

plt.subplot(427,sharex = axWhite)

hislat = categorySort[['Hispanic or Latino','Category']].sort_values(by = 'Hispanic or Latino', ascending = True).tail(5)

plt.barh(replaceString(hislat['Category']),hislat['Hispanic or Latino']/categorySort['Hispanic or Latino'].sum()*100)

plt.title('Top 5 Occupation Categories for Hispanic or Latinos(%)',fontSize = 20)

plt.xlabel('Employment%')



#Occupation-Related Wage for Hispanic or Latino(Right)

temp_hislat = []

for i in hislat['Category']:

    temp_hislat.append(wageMajor_n.loc[i.lower() ,'Annual Mean Wage'])

plt.subplot(428,sharex = axWhiteWage)

plt.barh(replaceString(hislat['Category']),temp_hislat,color=['Salmon'])

plt.title('Occupation-Related Wages for Hispanic or Latinos',fontSize = 20)

plt.xlabel('Wage in Dollar')





plt.subplots_adjust(wspace = 0.6,hspace = 0.4)

plt.show()
wageMajor_n['Annual Mean Wage'].mean()
topAnnualMeanWage = wageMajor.sort_values(by="Annual Mean Wage",ascending = False).head(30)

topAnnualMeanWage[['OCC_TITLE','Annual Medium Wage','Annual Mean Wage']].sort_values(by='Annual Mean Wage',ascending = True).plot(x = 'OCC_TITLE',kind = 'barh',figsize=(10,14))

plt.title('Wage Ranking by Occupation Categories in U.S ',fontsize = 20)

plt.xlabel('Wage (Dollar)',fontsize = 18)

plt.ylabel('Occupation Category Title',fontsize = 18)
eduRace = eduRace.rename(columns = {'Hispanic ':'Hispanic'})

eduRace
#Reindex the 'edu' table

edu = eduRace[['Detailed years of school','White','Black','Asian','Hispanic']]

edu.index = edu['Detailed years of school'].values

edu= edu[['White','Black','Asian','Hispanic']]

edu
#Combine Bechelor's degree, master's degree, doctorate degree to "Bachelor's degree and above"



edu.iloc[4:7].sum()

edu.loc["Bachelor's degree and above"] = edu.iloc[4:7].sum()

edu = edu.drop(edu.index[[4,5,6]])

edu
#Plot pie charts



colors = ['DodgerBlue','Salmon','LightSlateGray','SlateBlue','DarkTurquoise','HotPink','orange']

#education, whites

plt.figure(figsize = (10,10))

plt.subplot(221)

plt.pie(edu['White'],autopct='%1.1f%%',colors =colors)

plt.title('Education Level for Whites',fontSize = 15)



#education, blacks



plt.subplot(222)

plt.pie(edu['Black'],autopct='%1.1f%%',colors =colors)

plt.title('Education Level for Blacks',fontSize = 15)



#education, asians

plt.subplot(223)

plt.pie(edu['Asian'],autopct='%1.1f%%',colors =colors)

plt.title('Education Level for Asians',fontSize = 15)



#education, hispanic

plt.subplot(224)

plt.pie(edu['Hispanic'],autopct='%1.1f%%',colors =colors)

plt.title('Education Level for Hispanic and Latinos',fontSize = 15)

plt.legend(edu.index,bbox_to_anchor=(1.1, 0.5))

plt.show()