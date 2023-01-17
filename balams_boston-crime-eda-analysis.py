#Load Basic Libraries

import numpy as np

import pandas as pd



#EDA

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
#Load Dataset

data = pd.read_csv('../input/crime.csv', encoding = 'latin-1')

data.head(3)
#Basic understanding of data

basic = {'unique_value' : data.nunique(),

         'na_count' : data.isna().sum(),

        'Data_Type' : data.dtypes}

print('Shape of data is :',data.shape)

pd.DataFrame(basic)
data.drop(['INCIDENT_NUMBER','OFFENSE_CODE','OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION',

           'REPORTING_AREA','SHOOTING','STREET','Location'], 

          axis = 1, inplace = True)

data.head(3)
#Rename Column values

rename = {'DISTRICT' : 'District',

         'OCCURRED_ON_DATE' : 'Date',

         'YEAR' : 'Year',

         'MONTH' : 'Month',

         'DAY_OF_WEEK' : 'Week',

         'HOUR' : 'Hour',

         'UCR_PART' : 'Ucr_part'}

data.rename(index = str, columns= rename , inplace = True)

data.head(3)
#Convert Month index into str

data.Month.replace([1,2,3,4,5,6,7,8,9,10,11,12],

                  ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], inplace = True)



#Convert Month, Week, Hour, Ucr_part into Categorical variable

data.Month = pd.Categorical(data.Month, 

                            categories = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],

                            ordered = True)



data.Week = pd.Categorical(data.Week,

                           categories= ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], 

                           ordered = True)



data.Hour = pd.Categorical(data.Hour,

                          categories = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], ordered = True)



data.Ucr_part = pd.Categorical(data.Ucr_part,

                              categories = ['Part One','Part Two','Part Three','Other'], ordered = True)



data.dtypes
#Get Day_of_month from Date Column

data.Date = pd.to_datetime(data.Date)

data['Day_of_month'] = data.Date.dt.day

data.head(3)
#Handling Missing Values

#District

data.District.fillna(data.District.mode()[0], inplace = True)



#Ucr_part

data.Ucr_part.fillna(data.Ucr_part.mode()[0], inplace = True)
#Basic understanding of data

basic = {'unique_value' : data.nunique(),

         'na_count' : data.isna().sum(),

        'Data_Type' : data.dtypes}

print('Shape of data is :',data.shape)

pd.DataFrame(basic)
#sns.set(color_codes = True)

#sns.set_style('darkgrid')

#Distirct

plt.figure(figsize = (15,5))



plt.subplot(1,2,1)

sns.countplot( y = 'District', data = data, order = data.District.value_counts().index)

plt.ylabel('Count', size = 15)

plt.xlabel('District', size = 15)

plt.title('District vs CrimeRate', size = 20)



#Year

plt.subplot(1,2,2)

sns.countplot(x = 'Year', data =data, order = data.Year.value_counts().index)

plt.ylabel('CrimeRate', size = 15)

plt.xlabel('Year', size = 15 )

plt.title('Year vs CrimeRate', size = 20)



plt.show()



plt.figure(figsize = (15,15))



#Month

plt.subplot(2,2,1)

sns.countplot( x = 'Month', data = data)

plt.xticks(rotation = 45)

plt.ylabel('Count', size = 15)

plt.xlabel('Month', size = 15)

plt.title('Month vs CrimeRate', size = 20)



#Week

plt.subplot(2,2,2)

sns.countplot( x = 'Week', data = data)

plt.xticks(rotation = 45)

plt.ylabel('Count', size = 15)

plt.xlabel('Week', size = 15)

plt.title('Week vs CrimeRate', size = 20)



#Week

plt.subplot(2,2,3)

sns.countplot( x = 'Ucr_part', data = data, order = data.Ucr_part.value_counts().index)

plt.xticks(rotation = 45)

plt.ylabel('Count', size = 15)

plt.xlabel('Ucr_part', size = 15)

plt.title('Ucr_part vs CrimeRate', size = 20)



plt.show()



#Hour

plt.figure(figsize = (35,10))

plt.subplot(2,2,1)

sns.countplot( x = 'Hour', data = data)

plt.xticks(rotation = 45)

plt.ylabel('Count', size = 15)

plt.xlabel('Hour', size = 15)

plt.title('Hour vs CrimeRate', size = 20)
#Day_of_month

plt.figure(figsize = (35,10))

plt.subplot(2,2,1)

sns.countplot( x = 'Day_of_month', data = data)

plt.xticks(rotation = 45)

plt.ylabel('Count', size = 15)

plt.xlabel('Day_of_month', size = 15)

plt.title('Day_of_month vs CrimeRate', size = 20)