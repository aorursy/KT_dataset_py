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
#The dataset

import pandas as pd

BRFSS = pd.read_csv("../input/2015.csv")

BRFSS1 = BRFSS[['MENTHLTH','ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']]

BRFSS1.head(5)
#A variable to have the MENTHLTH in one place

BRFSS_M30= BRFSS['MENTHLTH']

BRFSS_M30.head(5)
#The 8 questions

BRFSS_8 = BRFSS[['ADPLEASR','ADDOWN','ADSLEEP','ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']]

BRFSS_8.head(8)
#Number of people that answer the question with the maximun of day in MENTHLTH

BRFSS_M= BRFSS1["MENTHLTH"].where(BRFSS1["MENTHLTH"]==30).count()

BRFSS_M
#Converting values from 77 and 99 into missing data | Replacing 88 with 0

import numpy as np

BRFSS_8.replace(77, np.NaN, inplace= True)

BRFSS_8.replace(99, np.NaN, inplace= True)

BRFSS_8.replace(88, 0, inplace= True)

#Making sure the changes where made

print("77 values\n",BRFSS_8.where(BRFSS_8==77).count())

print("99 values\n",BRFSS_8.where(BRFSS_8==99).count())
#Dropping out rows that has all values missing

BRFSS_8MD = BRFSS_8.dropna(how='all')

BRFSS_8MD
#Dropping rows where there is at least one missing data | In order to have a set of values to choose from

BRFSS_8DNAN= BRFSS_8.dropna()

BRFSS_8DNAN
#Imputating the values into the missing data with Hot Deck Imputation

#,'ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE'

for x in ['ADPLEASR', 'ADPLEASR','ADDOWN','ADSLEEP']:

    for i in BRFSS_8MD.index:

        if pd.isnull(BRFSS_8MD.loc[i, x]):

            BRFSS_V = np.random.choice(BRFSS_8DNAN[x])

            BRFSS_8MD.loc[i, x]=BRFSS_V

            

for x in ['ADENERGY','ADEAT1','ADFAIL','ADTHINK','ADMOVE']:

    for i in BRFSS_8MD.index:

        if pd.isnull(BRFSS_8MD.loc[i, x]):

            BRFSS_V = np.random.choice(BRFSS_8DNAN[x])

            BRFSS_8MD.loc[i, x]=BRFSS_V
#Making sure the datas were imputed. 

#I guess one of the issues with hot desk imputation choosing values with high probability to appear is that, 

#it will choose a lot of the values that correspond to zero

BRFSS_8MD
#Adding up the days and creating a column to save them

BRFSS_8MD['SUM']= BRFSS_8MD.sum(axis=1)

BRFSS_8MD
#To see how many values are in this condition

BRFSS_8MD_56= BRFSS_8MD['SUM'].where(BRFSS_8MD['SUM']>=56).count()

BRFSS_8MD_56
#Converging both the MENTHLTH and the 8 questions variables

BRFSS_2= BRFSS_8MD.join(BRFSS_M30)

BRFSS_2
BRFSS_2.isnull().sum()
#Just to compare

BRFSS_M=((BRFSS_2['MENTHLTH'].where(BRFSS_2['MENTHLTH']==30).count())/(BRFSS_M30.count())).mean().round(5) * 100

BRFSS_M
print(BRFSS_2['MENTHLTH'].where(BRFSS_2['MENTHLTH']==30).count())

print(BRFSS_2['MENTHLTH'].where(BRFSS_2['MENTHLTH']>=56).count())
x = len(BRFSS_2[(BRFSS_2['MENTHLTH']==30) & (BRFSS_2['SUM']>=56)])

x
y = len(BRFSS_2[(BRFSS_2['MENTHLTH']==30) & (BRFSS_2['SUM']<56)])

y
(x/(x+y))*100
x2 = len(BRFSS_2[(BRFSS_2['MENTHLTH']==30) & (BRFSS_2['SUM']>=40)])

x2
y2 = len(BRFSS_2[(BRFSS_2['MENTHLTH']==30) & (BRFSS_2['SUM']<40)])

y2
(x2/(x2+y2))*100