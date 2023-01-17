# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


PILOT_df=pd.read_excel('/kaggle/input/gas-stations-pilot-flying-j/locations PILOT FLYNG J Locations.xlsx')
PILOT_df.head(4)
#DEF= DIESEL EXHAUST FLUID 

#This is just for information, and applies to the 'BULK DEF LANES' column



#For importance of DEF, check the following article:

# https://www.cnet.com/roadshow/news/diesel-engine-exhaust-fluid-urea-scr-egr/





#CAT SCALE- Provides accurate weighing of vehicle to meet government regulations
PILOT_df.tail(4)
PILOT_df.shape   # 803 rows and 19 columns
PILOT_df.dtypes  #DataFrame Data types
PILOT_df.columns   # Column Names within PILOT_df  DataFrame
PILOT_df.describe()  #High level statistics
PILOT_df['Country'].unique()   #DataFrame contains store locations from two countries- USA (US) & CANADA (CA)
Store_Counts_USA=PILOT_df[PILOT_df['Country']=='US']

Store_Counts_USA.head(3)
len(Store_Counts_USA)  #NUMBER OF STORE LOCATIONS IN THE USA
Store_Counts_USA['State/Province'].value_counts()
Store_Counts_USA['State/Province'].value_counts().sum()  #cross-check to see if US stores sums up to 743.
Store_Counts_USA['State/Province'].value_counts(normalize=True)*100  
# Top 10 States with store locations

Store_dist=PILOT_df['State/Province'].value_counts()[0:10]  # Top 10

Store_dist
#Top 10 States account for 383 stores

Store_dist.sum()
# TOP 10 STATES BY STORE LOCATIONS

A=Store_Counts_USA['State/Province'].value_counts(normalize=True)*100

A[0:10]
# TOP 10 STATES ACCOUNT FOR ~ 51% OF ALL US PILOT FLYING J STORE LOCATIONS

A[0:10].sum()
Top_20_US=Store_Counts_USA['State/Province'].value_counts()[0:20]

Top_20_US
Stores_ValueCounts_df=pd.DataFrame(Top_20_US)

Stores_ValueCounts_df['Index']=Stores_ValueCounts_df.index

Stores_ValueCounts_df.columns

x=Stores_ValueCounts_df['Index']

y=Stores_ValueCounts_df['State/Province']

plt.figure(figsize=(20,8))

plt.bar(x,y,color='orange')

plt.xlabel('US STATE')

plt.ylabel('STORE LOCATIONS')

plt.title('Top 20 States by Store Locations in the USA')
Store_Counts_USA.head(5)
USA_States=Store_Counts_USA['State/Province'].unique()

USA_States
USA_States.shape
def STATE(x):

    ###  This function delivers the relevant state specific dataframe, store locations and analysis information for inputted state ###

   

        State_df=Store_Counts_USA[Store_Counts_USA['State/Province']==x]

        stores_list=State_df['City'].value_counts()

        stores_count=State_df['City'].value_counts().sum()

        print('STATE DATAFRAME OF:',x)

        print('________________________________________________________________________________________')

        print(State_df)

        print('________________________________________________________________________________________')

        print('TOTAL NUMBER OF STORE LOCATIONS IN:',x)

        print('________________________________________________________________________________________')

        print(stores_count)

        print('________________________________________________________________________________________')        

        print('LIST OF STORES IN:',x)

        print('________________________________________________________________________________________')

        print(stores_list)

        return

    

# Input State Code within function STATE to get the state specific dataframe, total store counts etc.

STATE('TN')

# For list of state codes, refer to USA_States array (that is listed above the STATE function)
# LOCATIONS WITH DIESEL LANES GREATER THAN 8, PARKING SPACES GREATER THAN 150 AND THAT CONTAINS A 'TACO' RESTAURANT

EE=PILOT_df[(PILOT_df['Diesel Lanes']>8) & (PILOT_df['Parking Spaces']>150) & (PILOT_df['Facilities/Restaurants'].str.contains('Taco'))]

EE
#For a sample list of unique restaurant names, see below (listing 20 names)

# Note, you can change the "Taco example in the above string argument to any other name from the array below"

PILOT_df['Facilities/Restaurants'].unique() [0:20]
#Creating a Texas specific DataFrame (Texasdf) below

Texasdf=PILOT_df[PILOT_df['State/Province']=='TX']

Texasdf['City'].unique()   #Locations of all sites in the State of Texas
# TEXAS Analysis with Cities containing greater than 10 Diesel Lanes with Parking Spaces atleast 100 or greater

Texasdf[(Texasdf['Diesel Lanes']>10) & (Texasdf['Parking Spaces']>=100)]

# We are first creating a Canada specific dataframe

CANADA=PILOT_df[PILOT_df['Country']=='CA']

#Multiple Condition Calgary Df(Calgary)

Calgary=CANADA[(CANADA['City']=='Calgary')&(CANADA['Parking Spaces']>100)&(CANADA['Facilities/Restaurants'].str.contains('Pizza'))]

Calgary
#State Specific Analysis: Example TENNESEE | Sort Top 10 Store Locations by number of Diesel Lanes (DESCENDING ORDER)

State_df2=Store_Counts_USA[Store_Counts_USA['State/Province']=='TN']

State_df2.sort_values(by='Diesel Lanes',ascending=False)[0:10]



#NOTE: You can easily do such analysis for other states. All you need to do to change the 'TN' parameter and insert the relevant state

# code listed in the USA_States array (see above)

#IF A GIVEN STATE DOES NOT HAVE AT LEAST 10 STORE LOCATIONS, SIMPLY ADJUST THE RANGE [0:10] LISTED ABOVE, OR YOU CAN REMOVE IT.

A_State=State_df2.sort_values(by='Diesel Lanes',ascending=False)[0:10]

x=A_State['City']

y=A_State['Diesel Lanes']



plt.figure(figsize=(20,8))

plt.bar(x,y,color='brown')

plt.xlabel('STORE LOCATIONS IN TENNESSEE')

plt.ylabel('DIESEL LANES')

plt.title('TOP STORE LOCATIONS BY NUMBER OF DIESEL LANES IN TENNESSEE')
#State Specific Analysis: Example TENNESEE | Sort Top 10 Store Locations by number of Parking Spaces (DESCENDING ORDER)

State_df2=Store_Counts_USA[Store_Counts_USA['State/Province']=='TN']

State_df2.sort_values(by='Parking Spaces',ascending=False)[0:10]



#NOTE: You can easily do such analysis for other states. All you need to do to change the 'TN' parameter and insert the relevant state

# code listed in the USA_States array (see above)

#IF A GIVEN STATE DOES NOT HAVE AT LEAST 10 STORE LOCATIONS, SIMPLY ADJUST THE RANGE [0:10] LISTED ABOVE, OR YOU CAN REMOVE IT.
A_State_Parking=State_df2.sort_values(by='Parking Spaces',ascending=False)[0:10]

x=A_State_Parking['City']

y=A_State_Parking['Parking Spaces']



plt.figure(figsize=(20,8))

plt.bar(x,y,color='olive')

plt.xlabel('STORE LOCATIONS IN TENNESSEE')

plt.ylabel('PARKING SPACES')

plt.title('TOP STORE LOCATIONS BY NUMBER OF PARKING SPACES IN TENNESSEE')
#State Specific Analysis: Example TENNESEE | Sort Top 10 Store Locations by number of Parking Spaces (DESCENDING ORDER)

State_df2=Store_Counts_USA[Store_Counts_USA['State/Province']=='TN']

State_df2.sort_values(by='Showers',ascending=False)[0:10]



#NOTE: You can easily do such analysis for other states. All you need to do to change the 'TN' parameter and insert the relevant state

# code listed in the USA_States array (see above)

#IF A GIVEN STATE DOES NOT HAVE AT LEAST 10 STORE LOCATIONS, SIMPLY ADJUST THE RANGE [0:10] LISTED ABOVE, OR YOU CAN REMOVE IT.
A_State_Showers=State_df2.sort_values(by='Showers',ascending=False)[0:10]

x=A_State_Showers['City']

y=A_State_Showers['Showers']



plt.figure(figsize=(20,8))

plt.bar(x,y,color='red')

plt.xlabel('STORE LOCATIONS IN TENNESSEE')

plt.ylabel('SHOWERS')

plt.title('TOP STORE LOCATIONS BY NUMBER OF SHOWERS IN TENNESSEE')
#State Specific Analysis: Example TENNESEE | Sort Top 10 Store Locations by number of Parking Spaces (DESCENDING ORDER)

State_df2=Store_Counts_USA[Store_Counts_USA['State/Province']=='TN']

State_df2.sort_values(by='Bulk DEF Lanes',ascending=False)[0:10]



#NOTE: You can easily do such analysis for other states. All you need to do to change the 'TN' parameter and insert the relevant state

# code listed in the USA_States array (see above)

#IF A GIVEN STATE DOES NOT HAVE AT LEAST 10 STORE LOCATIONS, SIMPLY ADJUST THE RANGE [0:10] LISTED ABOVE, OR YOU CAN REMOVE IT.
A_State_DEF=State_df2.sort_values(by='Bulk DEF Lanes',ascending=False)[0:10]

x=A_State_DEF['City']

y=A_State_DEF['Bulk DEF Lanes']



plt.figure(figsize=(20,8))

plt.bar(x,y,color='purple')

plt.xlabel('STORE LOCATIONS IN TENNESSEE')

plt.ylabel('BULK DEF LANES')

plt.title('TOP STORE LOCATIONS BY NUMBER OF BULK DEF LANES IN TENNESSEE')
# Creating a US TOP 25 Array (containing the Top 25 US States by Store Locations)



Top_25_US=Store_Counts_USA['State/Province'].value_counts()

Top_25_US

Top_25=pd.DataFrame(Top_25_US)

Top_25['State']=Top_25.index

Top_25.rename(columns={'State/Province':"Number of Locations"},inplace=True)

US_TOP25=Top_25.sort_values(by='Number of Locations',ascending=False)[0:25]

US_Top25_array=US_TOP25['State'].unique()

US_TOP25

US_Top25_array
# PLOTTING 25 DATA PLOTS SIMULTANEOUSLY FOR TOP 25 STATES, RANKED BY 'PARKING SPACES' AT VARIOUS CITY LOCATIONS IN THESE STATES

for a in US_Top25_array:

    DF=Store_Counts_USA[Store_Counts_USA['State/Province']==a]

    E=DF.sort_values(by='Parking Spaces',ascending=False)[0:10]

    x=E['City']

    y=E['Parking Spaces']

    plt.figure(figsize=(14,5))

    plt.bar(x,y,color='purple')

    plt.xlabel(a)

    plt.ylabel('Parking Spaces')

    plt.title(a)

    

    

    
# PLOTTING 25 DATA PLOTS SIMULTANEOUSLY FOR TOP 25 STATES, RANKED BY 'DIESEL LANES' AT VARIOUS CITY LOCATIONS IN THESE STATES

for a in US_Top25_array:

    DF=Store_Counts_USA[Store_Counts_USA['State/Province']==a]

    E=DF.sort_values(by='Diesel Lanes',ascending=False)[0:10]

    x=E['City']

    y=E['Diesel Lanes']

    plt.figure(figsize=(14,5))

    plt.bar(x,y,color='skyblue')

    plt.xlabel(a)

    plt.ylabel('Diesel Lanes')

    plt.title(a)

    
# PLOTTING 25 DATA PLOTS SIMULTANEOUSLY FOR TOP 25 STATES, RANKED BY 'SHOWERS' AT VARIOUS CITY LOCATIONS IN THESE STATES

for a in US_Top25_array:

    DF=Store_Counts_USA[Store_Counts_USA['State/Province']==a]

    E=DF.sort_values(by='Showers',ascending=False)[0:10]

    x=E['City']

    y=E['Showers']

    plt.figure(figsize=(14,5))

    plt.bar(x,y,color='pink')

    plt.xlabel(a)

    plt.ylabel('Showers')

    plt.title(a)
# PLOTTING 25 DATA PLOTS SIMULTANEOUSLY FOR TOP 25 STATES, RANKED BY 'BULK DEF LANES' AT VARIOUS CITY LOCATIONS IN THESE STATES

for a in US_Top25_array:

    DF=Store_Counts_USA[Store_Counts_USA['State/Province']==a]

    E=DF.sort_values(by='Bulk DEF Lanes',ascending=False)[0:10]

    x=E['City']

    y=E['Bulk DEF Lanes']

    plt.figure(figsize=(14,5))

    plt.bar(x,y,color='red')

    plt.xlabel(a)

    plt.ylabel('Bulk DEF Lanes')

    plt.title(a)
# CANADIAN STORES

CANADA=PILOT_df[PILOT_df['Country']=='CA']

CANADA  #Dataframe showing Canadian stores
CANADA.shape
CANADA.size
CANADA.count()  

#some of the columns such as 'Phone','Fax','Parking Spaces','Bulk DEF Lanes','Showers','Facilities/Restaurants' need to be updated by company

#to show latest data
len(PILOT_df[PILOT_df['Country']=='CA'])  # PILOT/FLYINGJ has 60 stores in Canada

CANADA['City'].unique()  #Unique store locations
CANADA['City'].value_counts()  #Distribution of Canadian stores by city
CANADA['City'].value_counts().sum()   #Total Number of Stores in CANADA. Data crosscheck to see if it adds up to 60 stores
#STATEWISE GROUPING OF STORE LOCATIONS  | EXAMPLE: OHIO (OH)

#NOTE: You can easily replace this analysis for any other state by replacing 'OH' below to the relevant state of interest.

#For list of state specific codes, refer to the USA_States (shown below)



STATE_group=PILOT_df.groupby(by='State/Province')

STATE_group.first()

len(STATE_group.get_group('OH'))  #number of locations in the State of OHIO



STATE_groupby=STATE_group.get_group('OH')

STATE_groupby['City']  # you can insert any of the column names from below to get the relevant information for this state

# for example, when you replace 'City' with "Parking Spaces", it will show the relevant data.



#COLUMN NAMES THAT CAN BE USED IN ABOVE GROUPBY STATE STORE LOCATIONS ANALYSIS

STATE_groupby.columns
#Use the relevant state codes within the above Groupby analysis for a given state

USA_States   # This is a ndarray
Groupby_inputs=PILOT_df['State/Province'].unique()   #this is now an array, where numpy operations can be handled

type(Groupby_inputs)

import numpy as np

Groupby_inputs.shape  #50 unique state/province options within the groupby listed above
type(Groupby_inputs)