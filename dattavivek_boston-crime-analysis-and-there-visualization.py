# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

os.chdir("../input")

os.listdir()



data = pd.read_csv("crime.csv",encoding = 'unicode_escape')

data.shape

# Any results you write to the current directory are saved as output.
#average crimes occuring every year, month and each day.

Yeardf=data.groupby('YEAR').size().reset_index(name='countperYr')

Yeardf['avgCrimes']=Yeardf.countperYr.map(lambda x: (x/12,x/365,x/(365*24)))

Yeardf[['CrimesPerYr','CrimesPerMnth','CrimesPerDay']]=pd.DataFrame(Yeardf['avgCrimes'].tolist())

Yeardf


#Number of crimes on Robbery/Burglary occured on all week days

robbery=data[data['OFFENSE_CODE_GROUP'].str.contains("Robbery|BURGLARY", na=False,case=False)].groupby(['DAY_OF_WEEK']).size().reset_index(name='count')

sns.barplot(x = "DAY_OF_WEEK",     

            y= "count",        

            data=robbery

            )
#Highest number of incidents for a particular crime occured in each year

data.groupby(['OFFENSE_CODE_GROUP','YEAR']).size().reset_index(name='count').sort_values(by=['count'], ascending=False)



#splitting the day into 'midnight','morning','afternoon','evening'

data['period']= pd.cut( data.HOUR,

                          [0,6,12,18,23],

                          labels=['midnight','morning','afternoon','evening'],

                          include_lowest=True

                          )

data['period']
#to find out the highest number of shootings occuring in a particular period.

shooting=data.loc[data['SHOOTING']=='Y'].groupby('period').size().reset_index(name='count')

sns.barplot(x = "period",     

            y= "count",        

            data=shooting

            )
#to find out the highest number of crimes occuring in a particular period.

period=data.groupby('period').size().reset_index(name='count')

sns.barplot(x = "period",     

            y= "count",        

            data=period

            )
#which district have less number of crimes

district=data.groupby(['DISTRICT']).size().reset_index(name='count')

sns.barplot(x = "DISTRICT",     

            y= "count",        

            data=district

            )


#what is the total number of crimes occured in each hour of a day

sns.countplot('HOUR',data= data)