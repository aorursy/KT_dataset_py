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

import matplotlib.pyplot as plt      # For base plotting

import seaborn as sns                # Easier plotting



# 1.4 Misc

import os

pd.options.display.max_columns = 200



print(os.listdir("../input"))

# 2.2 Read data file

data = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1")



# 2.3 Explore data

data.columns

data.dtypes                          # age is int64; This is a luxury. check np.iinfo('int64') and int8

np.iinfo('uint16')



# 2.3.1

data.describe()                      # set include = 'all' to see summary of 'object' types also

data.info()

data.shape                           # dim()

data.head()                          # head()

data.tail()





## District wise crime

plt.figure(figsize=(16,8))

data['DISTRICT'].value_counts().plot.bar()

plt.title('BOSTON: District wise Crimes')

plt.ylabel('Number of Crimes')

plt.xlabel('District')

plt.show()





#year wise crime trend

plt.figure(figsize=(16,8))

data['YEAR'].value_counts().plot.bar()

plt.title('BOSTON: Crimes - Yearly trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Year')

plt.show()

## Crime is increasing every year in Boston





#Offense code wise yearly trend 

 

plt.figure(figsize=(16,8))

data['OFFENSE_CODE_GROUP'].value_counts().plot.bar()

plt.title('BOSTON: OFFENSE_type trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Year')

plt.show()





########################################################

#Crime  weekly  trend 

 

plt.figure(figsize=(16,8))

data['DAY_OF_WEEK'].value_counts().plot.bar()

plt.title('BOSTON: Weekly  trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Week Day')

plt.show()





########################################################

#Crime  hourly   trend 

 

plt.figure(figsize=(16,8))

data['HOUR'].value_counts().plot.bar()

plt.title('BOSTON: Hourly  trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Hour of the day')

plt.show()

#crime drops in the night time. obvious observation

#midnight it is high



########################################################

###Boston Crimes: Year and  Police District Wise Crimes 

groups = data['DISTRICT'].unique()

n_groups = len(data['DISTRICT'].unique())-1



#fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.2

opacity= 0.8

plt.figure(figsize=(16,8))



df = data[['DISTRICT','YEAR']]



df_2015 = df.loc[(df['YEAR'] == 2015)]

df_2016 = df.loc[(df['YEAR'] == 2016)]

df_2017 = df.loc[(df['YEAR'] == 2017)]

df_2018 = df.loc[(df['YEAR'] == 2018)]



crimes_2015 = df_2015['DISTRICT'].value_counts()

crimes_2016 = df_2016['DISTRICT'].value_counts()

crimes_2017 = df_2017['DISTRICT'].value_counts()

crimes_2018 = df_2018['DISTRICT'].value_counts()



bar1 = plt.bar(index, crimes_2015, bar_width, alpha = opacity, color = 'b', label = '2015')

bar2 = plt.bar(index + bar_width, crimes_2016, bar_width, alpha = opacity, color = 'c', label = '2016')

bar3 = plt.bar(index+ bar_width+ bar_width, crimes_2017, bar_width, alpha = opacity, color = 'r', label = '2017')

bar4 = plt.bar(index+ bar_width+ bar_width+ bar_width, crimes_2018, bar_width, alpha = opacity, color = 'y', label = '2018')





plt.ylabel('Number of Crimes')

plt.xlabel("POLICE DISTRICTS")

plt.title('BOSTON: Police District Wise # of Yearly Crimes')

plt.xticks(index + bar_width, groups)

plt.legend()

#plt.savefig("C:\\Users\\bk42969\\Desktop\\BigData\\DistrictWiseYearlyCrimes.png")

plt.show()



########################################################



###Boston Crimes: Year and  crime Offense Code Wise Crimes 



#year wise crime trend

plt.figure(figsize=(16,8))

data['YEAR'].value_counts().plot.bar()

plt.title('BOSTON: Crimes - Yearly trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Year')

plt.show()

## Crime is increasing every year in Boston





#Offense code wise yearly trend 

 

plt.figure(figsize=(16,8))

data['OFFENSE_CODE_GROUP'].value_counts().plot.bar()

plt.title('BOSTON: OFFENSE_type trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Year')

plt.show()





########################################################

#Crime  weekly  trend 

 

plt.figure(figsize=(16,8))

data['DAY_OF_WEEK'].value_counts().plot.bar()

plt.title('BOSTON: Weekly  trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Week Day')

plt.show()





########################################################

#Crime  hourly   trend 

 

plt.figure(figsize=(16,8))

data['HOUR'].value_counts().plot.bar()

plt.title('BOSTON: Hourly  trend')

plt.ylabel('Number of Crimes')

plt.xlabel('Hour of the day')

plt.show()

#crime drops in the night time. obvious observation

#midnight it is high



########################################################

###Boston Crimes: Year and  Police District Wise Crimes 

groups = data['DISTRICT'].unique()

n_groups = len(data['DISTRICT'].unique())-1



#fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.2

opacity= 0.8

plt.figure(figsize=(16,8))



df = data[['DISTRICT','YEAR']]



df_2015 = df.loc[(df['YEAR'] == 2015)]

df_2016 = df.loc[(df['YEAR'] == 2016)]

df_2017 = df.loc[(df['YEAR'] == 2017)]

df_2018 = df.loc[(df['YEAR'] == 2018)]



crimes_2015 = df_2015['DISTRICT'].value_counts()

crimes_2016 = df_2016['DISTRICT'].value_counts()

crimes_2017 = df_2017['DISTRICT'].value_counts()

crimes_2018 = df_2018['DISTRICT'].value_counts()



bar1 = plt.bar(index, crimes_2015, bar_width, alpha = opacity, color = 'b', label = '2015')

bar2 = plt.bar(index + bar_width, crimes_2016, bar_width, alpha = opacity, color = 'c', label = '2016')

bar3 = plt.bar(index+ bar_width+ bar_width, crimes_2017, bar_width, alpha = opacity, color = 'r', label = '2017')

bar4 = plt.bar(index+ bar_width+ bar_width+ bar_width, crimes_2018, bar_width, alpha = opacity, color = 'y', label = '2018')





plt.ylabel('Number of Crimes')

plt.xlabel("POLICE DISTRICTS")

plt.title('BOSTON: Police District Wise # of Yearly Crimes')

plt.xticks(index + bar_width, groups)

plt.legend()

#plt.savefig("C:\\Users\\bk42969\\Desktop\\BigData\\DistrictWiseYearlyCrimes.png")

plt.show()



########################################################



###Boston Crimes: Year and  crime Offense Code Wise Crimes 

groups = data['OFFENSE_CODE_GROUP'].unique()

n_groups = len(data['OFFENSE_CODE_GROUP'].unique())-1



#fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.2

opacity= 0.8

plt.figure(figsize=(20,12))

# extract top 20 crimes of all the years

#crimes_total = df['OFFENSE_CODE_GROUP'].value_counts()

# pick first 20 rows

#crimes_total20 = crimes_total.iloc[:20]

n_groups= 19

index = np.arange(n_groups)



df = data[['OFFENSE_CODE_GROUP','YEAR']]

df1 = df.loc[df['OFFENSE_CODE_GROUP'].isin(['Motor Vehicle Accident Response' , 'Larceny','Medical Assistance' ,'Investigate Person','Other' ,'Drug Violation','Vandalism','Verbal Disputes','Towed','Investigate Property','Larceny From Motor Vehicle','Property Lost','Warrant Arrests','Aggravated Assault','Violations','Fraud','Residential Burglary','Missing Person Located','Auto Theft'])]



df_2015 = df1.loc[(df1['YEAR'] == 2015)]

df_2016 = df1.loc[(df1['YEAR'] == 2016)]

df_2017 = df1.loc[(df1['YEAR'] == 2017)]

df_2018 = df1.loc[(df1['YEAR'] == 2018)]



crimes_2015 = df_2015['OFFENSE_CODE_GROUP'].value_counts()

crimes_2016 = df_2016['OFFENSE_CODE_GROUP'].value_counts()

crimes_2017 = df_2017['OFFENSE_CODE_GROUP'].value_counts()

crimes_2018 = df_2018['OFFENSE_CODE_GROUP'].value_counts()



bar1 = plt.bar(index, crimes_2015, bar_width, alpha = opacity, color = 'b', label = '2015')

bar2 = plt.bar(index + bar_width, crimes_2016, bar_width, alpha = opacity, color = 'c', label = '2016')

bar3 = plt.bar(index+ bar_width+ bar_width, crimes_2017, bar_width, alpha = opacity, color = 'r', label = '2017')

bar4 = plt.bar(index+ bar_width+ bar_width+ bar_width, crimes_2018, bar_width, alpha = opacity, color = 'y', label = '2018')





plt.ylabel('Number of Crimes')

plt.xlabel("Offense type")

plt.title('BOSTON: Police Offense Type # of Yearly Crimes')

plt.xticks(index + bar_width, groups)

plt.legend()

plt.show()





#######################################################################



#Crime  hourly   trend 

 

plt.figure(figsize=(16,8))

data['SHOOTING'].value_counts().plot.bar()



groups = data['SHOOTING'].unique()

df = data[['SHOOTING','YEAR']]

df_Y = df.loc[(df['SHOOTING'] == 'Y')]

df_Y['SHOOTING'].value_counts()

df_Y['YEAR'].value_counts().plot.bar()



########################################################










