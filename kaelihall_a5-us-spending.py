# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fs=pd.read_csv('/kaggle/input/usa-federal-spending-in-q4-2019/FY2019Q1-Q4_All_FA_AccountBalances_2020-02-08_H21M39S22_1.csv')

fs#Opening and reading the raw dataframe
fs.describe()#Just investigating
fs.index#Looking at how long the list is in order to decide how much I need to narrow this data down
cleaned=fs[['agency_name','gross_outlay_amount','total_budgetary_resources']]#I want to narrow this data down to just the name of the agency, outlay and budget

cleaned
values=cleaned['agency_name'].value_counts().head(20)#There are a lot of agencies, so I used this function to narrow it down to the top 20 agencies with the most submissions in the time period

values

values.plot(kind='bar')#I also wanted to visualize the differences between how many submissions were recorded per agency
cleaned.describe()#More investigating
cleaned.sort_values('agency_name')#More investigating
deptdef = cleaned['agency_name'] == 'Department of Defense'#Storing the booleans for whether an agency name includes "Department of Defense"

DoDdataframe = cleaned[deptdef]#Applying that new series of booleans to the dataframe to create a new dataframe which only includes info from the "Department of Defense"

DoDdataframe
DoDdataframe.describe()#Investigating that particular dataframe
dodtot=DoDdataframe.cumsum()#Comparing the cumulative sum of the budget vs outlay on that particular dataframe

dodtot.plot()#Plotting that out on a line graph. 
depttran = cleaned['agency_name'] == 'Department of Transportation'#Trying the same formula out for a different agency name

DoTdataframe = cleaned[depttran]

DoTdataframe

#It worked!
DoTdataframe.describe()
dottot=DoTdataframe.cumsum()

dottot.plot()#Same problem with the visualization here
depted = cleaned['agency_name'] == 'Department of Education'#Trying again, still works

DoEdataframe = cleaned[depted]

DoEdataframe
doetot=DoEdataframe.sum()

#gross_outlay_amount=float(gross_outlay_amount)

#total_budgetary_resources=float([total_budgetary_resources])

print(doetot)

doetot[['gross_outlay_amount','total_budgetary_resources']].plot(kind='bar')
cleaned['agency_name']

type(cleaned['agency_name'])#This is helping me figure out how to write the code so that I understand better what my existing data is stored as
#Now that I have discovered how to segment out each agency name and their totals, I want to make a function that can do this for the top 20. 

agencies=values.keys()#I am producing a variable to store the list of all of the agency names



def totspend(dept):#Creating a procedure that does the same thing as my formulas above to extract info for various agency names

    deptx = cleaned['agency_name'] == dept#Creating variables for departments instead of the manual inputs

    dataframex = cleaned[deptx]#Creating a variable for new dataframes

    totals=dataframex.sum()#Creating a variable to store the sums in

    totals[0]=dept#I don't want the sum of the 0th position because that is the agency_name which is a string, and it will repeat itself over and over again. Since the agency name is by definition the variable "dept" I am programming that variable to replace that 0th place sum. 

    return totals



visual=pd.DataFrame()#Creating a new dataframe to store the results of my new function



for theagencies in agencies:#Looping the list of agencies I created above from values.keys()

    finals=totspend(theagencies)#Looping the list of agencies through my function

    #print(finals)

    visual=visual.append(finals,ignore_index=True)#Adding the new information to a new dataframe. 

    

    #finals.DataFrame(columns("agency_name","gross_outlay_amount","total_budgetary_resources"))

print(visual)





visual.plot(kind='bar',x="agency_name",figsize=(15,10))

#Plotting all of these agencies and their spend/budget 


mil=pd.DataFrame()

allmilitary=['Department of the Navy','Department of Defense','Department of the Army','Department of Veterans Affairs','Department of the Air Force']#Creating a list of all of those agencies

for military in allmilitary:#Loop that list through my function

    militarymoney=totspend(military)

    mil=mil.append(militarymoney,ignore_index=True)#appending my new list to my new Dataframe

All_Military=mil.sum()#Creating a series which includes my new information



visual=visual.append(All_Military,ignore_index=True)#Appending my new information to my old dataframe

visual.plot(kind='bar',x="agency_name",figsize=(15,10))#Replotting it so I can see how it compares. That's interesting!