# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import zipfile



    

from subprocess import check_output

print(check_output(["ls", "../input/lcrime_z"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
all_crime = pd.read_csv('../input/lcrime_z/louisville_crime2003to2017.csv',index_col=None, header=0)
#average time to report crime in hours, by crime type



gdf = pd.DataFrame({'Mean Hours To Report Crime' : all_crime.groupby('CRIME_TYPE').mean()['Time To Report in Days']*24})

gdf.plot.bar()

plt.show()
#plot # of crms committed over time, by crime types



#crimes = ['ASSAULT','BURGLARY','DRUGS/ALCOHOL VIOLATIONS','FRAUD','MOTOR VEHICLE THEFT','OTHER','THEFT/LARCENY','VANDALISM','VEHICLE BREAK-IN/THEFT', 'HOMICIDE', 'SEX CRIMES', 'ARSON', 'ROBBERY', 'WEAPONS']

crimes = ['ASSAULT','BURGLARY','DRUGS/ALCOHOL VIOLATIONS','FRAUD','MOTOR VEHICLE THEFT','OTHER','THEFT/LARCENY','VANDALISM','VEHICLE BREAK-IN/THEFT']

all_crime_a = all_crime.drop(all_crime[all_crime.YEAR_OCCURED < 2004].index)



#here, you can include which crimes specificly that you want plotted

all_crime_a = all_crime_a[(all_crime_a['CRIME_TYPE'].isin(crimes))]



fig, ax = plt.subplots(figsize=(10, 6))



all_crime_a.groupby(['YEAR_OCCURED', 'CRIME_TYPE']).count()['INCIDENT_NUMBER'].unstack().plot(ax=ax)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()