# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
population=pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')
reactors=pd.read_excel('/kaggle/input/reactors/usreact17.xlsx')
population.head()
#[:8]
reactors.head()
#syntax refers to rows 0-8 ([8:] would refer to all rows after row 8)
reactors[:8]
reactors[4:]
#get rid of empty rows
df=reactors[4:]

#iterate over every column
for column in df:
    
    #rename each column as the name provided in row 4
    df = df.rename(columns = {column:df[column][4]}) 
df
#can get rid of original row that had column headers
df=df[1:]
df.head()
df.State.value_counts()
#give df a new name so that we can distinguish between the dataframes with and
#without state totals
short_df=df
#df.State.value_counts() will give us a series of state names and the number of
#time they show up in the 'State' column.

#Adding the .index will give a list-type structure such that we can iterate through
#every state
df.State.value_counts().index
#each entry is a string
type(df.State.value_counts().index[0])
#iterate through every unique string in the 'State' column
for string in df.State.value_counts().index:
    
    #can check if the phrase 'Total' is in the entry (this will be read as true or false)
    if 'Total' in string:
        #if true
        
        #datframe of true/false based on whether the string is in the 'State' column
        false_df=short_df['State'].isin([string])
        
        #dataframe with the falses (only states that don't have 'Total' in their name)
        short_df=short_df[-false_df]
short_df
#check work
short_df.State.value_counts()
alabama=short_df[short_df.State.isin(['AL'])].reset_index()
sum=0
for row in range(len(alabama)):
    sum+=alabama['January'][row]
sum
#simpler method
alabama.sum()['January']
short_df[short_df.State.isin(['AL'])]
short_df[short_df.State.isin(['AL'])].reset_index()
df[df['State'].isin(['Alabama Total'])]['January']
#this will require a couple different 'levels' to break down

#first let's create a loop that will go through each of the desired states
midwest=df[df['State'].isin(['MN','IL','IN','WI','OH'])].reset_index()
sum=0
for row in range(len(midwest)):
    sum+=midwest['Year_to_Date'][row]
sum
#simpler method
midwest.sum()['Year_to_Date']
midwest.sum()
#add sun belt as a region for comparison
sunbelt=df[df['State'].isin(['FL','GA','SC','AL','MS'])].reset_index()
#data for summer months
raw_data={'Month':['June','June','July','July','August','August'],
         'Region':['Midwest','Sun Belt','Midwest','Sun Belt','Midwest','Sun Belt'],
         'Total':[midwest.sum()['June'],sunbelt.sum()['June'],midwest.sum()['July'],sunbelt.sum()['July'],
                 midwest.sum()['August'],sunbelt.sum()['August']]}
summer=pd.DataFrame(raw_data, columns=['Month','Region','Total'])

summer
import matplotlib.pyplot as plt
#easiest way for me to do stacked bar plot is to add each level one by one
#for each level, specify x axis and y axis data

p1=plt.bar(summer[summer['Region'].isin(['Midwest'])]['Month'], summer[summer['Region'].isin(['Midwest'])]['Total'])
p2=plt.bar(summer[summer['Region'].isin(['Sun Belt'])]['Month'], summer[summer['Region'].isin(['Sun Belt'])]['Total'],
       bottom=summer[summer['Region'].isin(['Midwest'])]['Total'])
plt.ylabel('Energy Output (MegaWatt Hours)')
plt.title('Energy Output for Summer Months')
plt.legend((p1[0], p2[0]), ('Midwest', 'Sun Belt'))
#these variables are created beforehand so that the plotting code is a bit easier to interpret
#we want y axis data to be specified by month
#this will by default have different data for each region

june_totals=summer[summer['Month'].isin(['June'])]['Total']
july_totals=summer[summer['Month'].isin(['July'])]['Total']
august_totals=summer[summer['Month'].isin(['August'])]['Total']
#on the x axis we want each region, but plotted by month
    #summer[summer['Month'].isin(['June'])]['Region'] 
    #the above code isolates all June data from the dataframe, then adding the ['Region'] column we specify that
    #we want the x axis to be separated by region

p1=plt.bar(summer[summer['Month'].isin(['June'])]['Region'], june_totals)
p2=plt.bar(summer[summer['Month'].isin(['July'])]['Region'], july_totals,
       bottom=june_totals)

#pandas wouldn't compute (june_totals + july_totals) due to indexing/formatting, but it works when I add .values
p3=plt.bar(summer[summer['Month'].isin(['August'])]['Region'], august_totals,
          bottom=june_totals.values + july_totals.values)
           
plt.ylabel('Energy Output (MegaWatt Hours)')
plt.legend((p1[0], p2[0],p3[0]), ('June', 'July','August'))
il=short_df[short_df['State'].isin(['IL'])]
il
il_short=il[['Plant ID','Plant Name','January','February','March']]
p1=plt.bar(il_short['Plant Name'],il_short['January'])
p2=plt.bar(il_short['Plant Name'],il_short['February'],bottom=il_short['January'])

plt.xticks(rotation=90)
