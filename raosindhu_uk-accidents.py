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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



acc_05_07 = pd.read_csv("../input/accidents_2005_to_2007.csv", index_col='Accident_Index', parse_dates=True)

acc_09_11 = pd.read_csv("../input/accidents_2009_to_2011.csv", index_col='Accident_Index', parse_dates=True)

acc_12_14 = pd.read_csv("../input/accidents_2012_to_2014.csv", index_col='Accident_Index', parse_dates=True)



acc_05_07.head()

# acc_09_11.head()

# acc_12_14.head()



# check if the columns are same in both data sets

print(acc_05_07.columns.equals(acc_09_11.columns)) 

print(acc_05_07.columns.equals(acc_12_14.columns))



# combine 3 data sets with accidents info into a single dataset

acc_05_14 = pd.concat([acc_05_07, acc_09_11, acc_12_14])

# acc_05_14.head()

# acc_05_14.info()



#concatinating date and time columns

acc_05_14['date_time'] = acc_05_14['Date']+ ' ' + acc_05_14['Time']

# acc_05_14.head()

# acc_05_14.info()



#converting into date time format

time_format = '%d/%m/%Y %H:%M'

acc_05_14['date_time'] = pd.to_datetime(acc_05_14['date_time'], format=time_format)

# acc_05_14.head()

# acc_05_14.info()



#removing column will allmissing values(no data) and redundant columns

acc_05_14.drop(columns=['Junction_Detail'], inplace=True) #Delete columns without reassigning to dataframe

acc_05_14.head()

# acc_05_14.info()



#acc_05_14['date_time'].describe()



weekday = {1:'Sunday',

           2:'Monday',

           3:'Tuesday',

           4:'Wednesday',

           5:'Thursday',

           6:'Friday',

           7:'Saturday'}

print(weekday)



acc_05_14['Month'] = acc_05_14['date_time'].dt.month



#number of accidents each year

# acc_by_year = acc_05_14.groupby('Year').size()

# acc_by_year

acc_05_14.groupby('Year').size().plot(kind='bar', title='Accidents by Year')

plt.ylabel('Number of Accidents') 



#number of accidents each month of the year

# acc_by_month = acc_05_14.groupby('Month').size()

# acc_by_month

acc_05_14.groupby('Month').size().plot(kind='bar', title='Accidents by Month')

plt.ylabel('Number of Accidents') 



#number of accidents Time of the Day

acc_05_14.groupby(acc_05_14['date_time'].dt.hour).size().plot(kind='bar', title='Accidents by Hour')

plt.ylabel('Number of Accidents') 

plt.xlabel('Hour of the Day') 



#number of accidents each month and year

#using pivottable for charts

pt_month_year = acc_05_14.pivot_table(index='Month', columns = 'Year', values = 'date_time', aggfunc = 'count')

pt_month_year

pt_month_year.plot(xticks=[1,2,3,4,5,6,7,8,9,10,11,12])

plt.title('Accidents by Month and Year')

plt.xlabel('Month') 

plt.ylabel('Number of Accidents') 

#end of using pivottable



#Number of Casualities

acc_05_14['Number_of_Casualties'].unique()

def group_casualty(x):

    if (x >= 5):

        return '5+'

    else: 

        return x

acc_05_14['Casualties'] = acc_05_14['Number_of_Casualties'].apply(group_casualty)

acc_05_14['Casualties'].describe()



acc_05_14.groupby(['Casualties']).size().plot(kind='bar')

plt.title('Casualties')

plt.xlabel('Number of Casualties per accident') 



#Accident Severity

acc_05_14['Accident_Severity'].unique()

# x_labels = acc_05_14.groupby('Accident_Severity').size()

acc_05_14.groupby('Accident_Severity').size().plot( kind = 'bar')

plt.title('Accidents Severity')

plt.ylabel('Number of Accidents') 

# plt.set_xticklabels(x_labels)# adding data labels



#Light Conditions

acc_05_14['Light_Conditions'].unique()

acc_05_14.groupby('Light_Conditions').size()

acc_05_14.groupby('Light_Conditions').size().plot(kind = 'pie', autopct='%1.0f%%')

plt.title('Light Conditions')

plt.ylabel('Number of Accidents') 



#split the light condition column

new = acc_05_14['Light_Conditions'].str.split(':', n = 1, expand = True)

acc_05_14['DayCondition'] = new[0]

print(acc_05_14.groupby('LightCondition').size())

acc_05_14['LightCondition'] = new[1]

acc_05_14.groupby('LightCondition').size().plot(kind = 'pie', autopct='%1.0f%%')



#correcting the darkness from darkeness

acc_05_14.loc[acc_05_14['DayCondition'] == 'Darkeness','DayCondition'] = 'Darkness'    

print(acc_05_14.groupby('DayCondition').size())



acc_05_14.pivot_table(index='LightCondition', columns ='DayCondition', 

                      values = 'date_time', aggfunc = 'count').plot(kind = 'bar')



# type(acc_05_14[['Light_Conditions']])

#Weather Conditions

acc_05_14['Weather_Conditions'].unique()

acc_05_14.groupby('Weather_Conditions').size().plot(kind = 'bar')

plt.title('Weather Conditions')

plt.ylabel('Number of Accidents') 
