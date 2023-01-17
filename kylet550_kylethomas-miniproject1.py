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
import pandas as pd

import matplotlib.pyplot as plt



#Opening the six data file

accidents = pd.read_csv("../input/whitewater-accidents/accidents.csv")

factors = pd.read_csv("../input/whitewater-accidents/factors.csv")

injuries = pd.read_csv("../input/whitewater-accidents/injuries.csv")

causes = pd.read_csv("../input/whitewater-accidents/causes.csv")

accidents_factors = pd.read_csv("../input/whitewater-injuries-and-factors/accidents_factors.csv")

accidents_injuries = pd.read_csv("../input/whitewater-injuries-and-factors/accidents_injuries.csv")

#Viewing first 10 rows

accidents.head(10)
#Viewing last 10 rows

accidents.tail(10)


#Printing the list of column headers

print(accidents.columns.values)
#Removing columns containing duplicate data from the Accidents dataset

accidents = accidents.drop(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25'], axis=1)



#Removing personally identifiable information (PII) columns

accidents = accidents.drop(['victimname', 'contactname', 'contactphone', 'contactemail', 'othervictimnames', 'description'], axis=1)



#Removing unhelpful content

accidents = accidents.drop(['groupinfo'], axis=1)
#Renaming columns to more readable.  Update dataframe with new column names. Print accidents data to test it. 

accidents = accidents.rename(columns = {'accidentdate':'Date', 'countryabbr':'Country', 'state':'State', 'river':'River', 'section':'Section', 'location':'Location',

                                       'waterlevel':'Water Level', 'rellevel':'Relative Level','difficulty':'Difficulty', 'age':'Age', 'experience':'Experience',

                                       'privcomm':'Private or Commercial', 'boattype':'Boat Type', 'numvictims':'Number of Victims', 'cause':'Cause','status':'Status'})

accidents
#Leverage the length of the data frame to determine the total count of accidents

print(str(len(accidents)) + " total accidents recorded in dataset")



#Filtering table down to values specifically for Washington

washington = accidents[(accidents.State == 'WA')]



#Calculating how many accidents in the dataset occurred in Washington State

total_rows_washington = washington.shape[0]

print(str(total_rows_washington) + " accidents occurred in Washington")
#Aggregating the counts of accidents in U.S. States

stateCounts = accidents['State'].value_counts()



#Print out the top 10

stateCounts.head(10)
#Splitting Date column into individual Year, Month, and Day columns

accidents[['Year','Month', 'Day']] = accidents.Date.str.split("-",expand=True)



#Testing split

accidents
#Dropping the NaN Age values from the data frame before plotting age distribution

x = accidents.dropna()



#Dropping the 0 Age values from the data frame before plotting age distribution

x = x[x.Age != 0]



#Print the mean Age in the data frame

print("The mean age involved in an accident is " + str(x['Age'].mean()) + " years old")



#Print the median age in the data frame

print("The median age involved in an accident is " + str(x['Age'].median()) + " years old")



#Creating a histogram to show distribution of ages of people involved in incidents

x.hist(['Age'], bins=20)



#Add a chart title and axis labels

plt.title('Distribution of Age Involved in Whitewater Accidents')

plt.xlabel('Age')

plt.ylabel('Frequency')

#Pulling all unique values in Boat Type column

accidents['Boat Type'].unique()



#I needed to do manual queries on American Whitewater website to determine what these letters actually referred to.  

#Nothing was found in the uploaded data.
#Replacing confusing single letter labels with Inflatable Kayak, Kayak - Other, Open Canoe, Other, Raft, Whitewater Kayak

accidents['Boat Type'] = accidents['Boat Type'].replace('K', 'Kayak')

accidents['Boat Type'] = accidents['Boat Type'].replace('R', 'Raft')

accidents['Boat Type'] = accidents['Boat Type'].replace('O', 'Other')

accidents['Boat Type'] = accidents['Boat Type'].replace('N', 'Open Canoe')

accidents['Boat Type'] = accidents['Boat Type'].replace('T', 'Kayak - Other')

accidents['Boat Type'] = accidents['Boat Type'].replace('I', 'Inflatable Kayak')



#Testing it out

accidents
#Looking up specific accident ID's and comparing with searches from https://www.americanwhitewater.org/content/Accident/view/

#Attempting to learn what unique values for Boat Type mean.  Very manual (and annoying) task!

accidents.loc[accidents['id'] == 62905]
#Aggregating the counts of different boat types involved in accidents

boatCounts = accidents['Boat Type'].value_counts()



#Printing it out

print(boatCounts)

#Sorting all of the data by the boatCounts values with highest being on top, lowest on bottom

boatCounts = boatCounts.sort_values() 



#Creating a horizontal bar chart of the counts of different boat types involved in accidents

boatCounts.plot.barh()



#Adding a chart title and axis labels

plt.title('Whitewater Accidents by Boat Type')

plt.xlabel('Number of Accidents')

plt.ylabel('Boat Type')
#Pulling all the unique values for Relative Level from the data frame.  This will be used to replace with meaningful labels.

accidents['Relative Level'].unique()
#Replacing confusing single letter labels with Low, Medium, High, Flood

accidents['Relative Level'] = accidents['Relative Level'].replace('L', 'Low')

accidents['Relative Level'] = accidents['Relative Level'].replace('M', 'Medium')

accidents['Relative Level'] = accidents['Relative Level'].replace('H', 'High')

accidents['Relative Level'] = accidents['Relative Level'].replace('F', 'Flood')



#Testing it out

accidents
#Aggregating the counts of different river levels involved in accidents

levelCounts = accidents['Relative Level'].value_counts()



print(levelCounts)
#Sorting all of the data by the levelCounts values with highest being on top, lowest on bottom

levelCounts = levelCounts.sort_values() 



#Creating a horizontal bar chart of the counts of different river levels involved in accidents

levelCounts.plot.barh()



#Adding chart title and axis labels

plt.title('Whitewater Accidents by River Level')

plt.xlabel('Number of Accidents')

plt.ylabel('River Level')
accidents['Experience'].unique()
#Replacing confusing single letter labels with Expert, Extensive, Inexperienced, Some

accidents['Experience'] = accidents['Experience'].replace('X', 'Expert')

accidents['Experience'] = accidents['Experience'].replace('E', 'Extensive')

accidents['Experience'] = accidents['Experience'].replace('I', 'Inexperienced')

accidents['Experience'] = accidents['Experience'].replace('S', 'Some')



#Testing it out

accidents

#Dropping NaN values from data frame

y = accidents.dropna()



#Dropping " " values from data frame

y = y[y.Experience != " "]



#Aggregating the counts of different experience levels involved in accidents

experienceCounts = y['Experience'].value_counts()



print(experienceCounts)
#Sorting all of the data by the experienceCounts values with highest being on top, lowest on bottom

experienceCounts = experienceCounts.sort_values() 



#Creating a horizontal bar chart of the counts of different experience levels involved in accidents

experienceCounts.plot(kind='barh')



#Adding chart title and axis labels

plt.title('Whitewater Accidents by Experience Level')

plt.xlabel('Number of Accidents')

plt.ylabel('Experience Level')
#Aggregating the counts accidents per year

yearCounts = accidents['Year'].value_counts()



print(yearCounts)

#Sorting all of the data by the yearCounts values with highest being on top, lowest on bottom

yearCounts = yearCounts.sort_index() 



#Plotting a bar chart of data

yearCounts.plot(kind='bar')



#Add a chart title and axis labels

plt.title('Documented Whitewater Accidents over the Years 1924-2019')

plt.xlabel('Year')

plt.ylabel('Number of Accidents')
#Previewing causes.csv data file

causes
#Utilize the causes.csv file to replace meaningless values in Cause column in accidents.csv file with meaningful labels



#Iterating through each ID in the causes data file

for i in causes['id']:

    

    #Iterating through each numeric and meaningless Cause in accidents data frame

    for j in accidents['Cause']:

        

        #If the two values match, the meaningful string value should replace the meaningless numeric value

        if j == i:

            

            #Accessing the meaningful string from the causes data file

            var = causes.loc[causes['id'] == i, 'cause'].iloc[0]

            

            #Replacing the meaningless numeric value for cause with the meaningful string value

            accidents['Cause'] = accidents['Cause'].replace(j, var)



#Testing it out

accidents
#Aggregating the counts of different causes for accidents

causeCounts = accidents['Cause'].value_counts()



print(causeCounts)
#Sorting all of the data by the caseCounts values with highest being on top, lowest on bottom

causeCounts = causeCounts.sort_values() 



#Creating a bar chart of the counts of different causes for accidents

causeCounts.plot.barh()



#Add chart title and axis labels

plt.title('Causes of Whitewater Accidents')

plt.xlabel('Number of Accidents')

plt.ylabel('Cause of Accident')
#Previewing the accidents_factors data file

accidents_factors
#Previewing the factors data file

factors
#Utilize the factors.csv file to replace meaningless values in factor_id column in accidents_factors.csv file with meaningful labels



#Iterating through each ID in the factors data file

for i in factors['id']:

    

    #Iterating through each numeric and meaningless Factor in accidents_factor data file

    for j in accidents_factors['factor_id']:

        

        #If the two values match, the meaningful string value should replace the meaningless numeric value

        if j == i:

            

            #Accessing the meaningful string from the factors data file

            var = factors.loc[factors['id'] == i, 'factor'].iloc[0]

            

            #Replacing the meaningless numeric value for factors with the meaningful string value

            accidents_factors['factor_id'] = accidents_factors['factor_id'].replace(j, var)



#Testing it out

accidents_factors
#Aggregating the counts of different factors for accidents

factorCounts = accidents_factors['factor_id'].value_counts()



print(factorCounts)
#Sorting all of the data by the factorCounts values with highest being on top, lowest on bottom

factorCounts = factorCounts.sort_values() 



#Creating a bar chart of the counts of different factors for accidents

factorCounts.plot.barh()



#Adding chart title and axis labels

plt.title('Factors of Whitewater Accidents')

plt.xlabel('Number of Accidents')

plt.ylabel('Factors of Accident')
#Previewing injuries data file

injuries
#Previewing accidents_injuries data file

accidents_injuries
#Utilize the factors.csv file to replace meaningless values in factor_id column in accidents_factors.csv file with meaningful labels



#Iterating through each ID in the injuries data file

for i in injuries['id']:

    

    #Iterating through each numeric and meaningless Injury in accidents_injuries data file

    for j in accidents_injuries['injury_id']:

        

        #If the two values match, the meaningful string value should replace the meaningless numeric value

        if j == i:

            

            #Accessing the meaningful string from the injuries data file

            var = injuries.loc[injuries['id'] == i, 'injury'].iloc[0]

            

            #Replacing the meaningless numeric value for injuries with the meaningful string value

            accidents_injuries['injury_id'] = accidents_injuries['injury_id'].replace(j, var)



#Testing it out

accidents_injuries
#Aggregating the counts of different injuries for accidents

injuryCounts = accidents_injuries['injury_id'].value_counts()



print(injuryCounts)
#Sorting all of the data by the injuryCounts values with highest being on top, lowest on bottom

injuryCounts = injuryCounts.sort_values() 



#Creating a bar chart of the counts of different factors for accidents

injuryCounts.plot.barh()



#Adding chart title and axis labels

plt.title('Injuries of Whitewater Accidents')

plt.xlabel('Number of Accidents')

plt.ylabel('Injuries in Accident')