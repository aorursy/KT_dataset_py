#import key python libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
A=pd.read_csv("C:\Python_Data\KAGGLE\Trips_by_Distance.csv")  #reading the input csv file
A.head(5)   #checking the first 5 rows of the DataFrame, A.  You can insert any integer variable to see the number of rows.
A.dtypes   #DataFrame datatypes
A.columns  #DataFrame Column names
A.describe()  #High level DataFrame statistics
A.ndim
A.shape  #number of rows and columns
#Converting Date to Pandas Datetime format
A['Date']=pd.to_datetime(A['Date'])
A.info()  #check to see if above operation worked.
# MULTIPLE CONDITIONAL SLICING OF DATAFRAME A

B=A[(A['State Postal Code']=='TN') & (A['Date']== '2020/07/06')].dropna()
# COMMENTS ON SLICED DATAFRAME B:
#replace  the state abbreviation and the sliced DataFrame,B, will represent the chosen state of interest
#B is a sliced dataframe for a given state for the week of July 6th, 2020. You can change this date for any other weekly period for analysis
#dropna()  was used to remove the State level summary data, to accurately depict the number of counties by a given state
B.describe()  #Summary Statistics for sliced DataFrame, B. In this case it is shown for the State of Tennessee

#  EXAMPLE- Sorting (Ascending Order) Top 50 counties by" Population Not Staying at Home within the State of TN
# Note, you can replace TN by any other State Postal Code (see above) to generate similar data for other states

B.sort_values(by=['Population Not Staying at Home'],ascending=True)[0:50]

## SORTING COUNTIES WITHIN A GIVEN STATE IN ASCENDING ORDER
## By: Population Not Staying at Home  (example is shown below for TN)
B.sort_values(by=['Population Not Staying at Home'],ascending=False)[0:50]
B.shape
B['Number of Trips'].sum() #Total sum of trips made during the week of 2020/07/06
#Example Data for State of Tennessee
B.describe()
!pip install sidetable
import sidetable
E=A[A['Date']=='2020/07/06'].dropna()    #E represents sliced dataframe representing data for the week of 2020/07/06
# dropna was used to remove state and national summarization data, 
# to get an accurate count of counties by given State Postal Code.
E.stb.freq(['State Postal Code'])  #This is the cummulative data for the week of 2020/07/06
#similar analysis can be done for various weeks of 2020 or 2019 (for example)
def  State(x,y):
    ### Generates State level customized DataFrame for a given state (showing all counties) and timeframe ###
    
    DF=A[(A['State Postal Code']==x) & (A['Date']== y)].dropna()
    print(DF.head(20))
    return

# x= State code, from "State Postal COde " Column, for example TN represents Tennessee, CA represents California
#y= Time (in Pandas Datetime format): example YEAR/Month/Date that are listed within the 'Date' column of DataFrame A
x='CA'   # replace CA with any other state code to generate that state's county data
y='2020/07/06'  #replace this date with any other given date from the 'date' column of DataFrame A
State(x,y)    #State is the function that generates the State level county data for any given 
#PLOTTING NATIONAL USA DATA
Xnational=A[(A['Level']=='National') & (A['Date']=='2020/09/15')].transpose()[9:]
Xnational
Xnational.reset_index(inplace=True)
Xnational.columns=['Trip_Type',"Trip_Number"]
Xnational
#MATPLOTLIB DATA PLOTS CODE SPECIFICATIONS:  USA NATIONAL DATA
plt.figure(figsize=(20,10))
axes = plt.gca()
plt.xticks(rotation=45)
plt.bar(Xnational.Trip_Type,Xnational.Trip_Number,color='green')
plt.xlabel('Type of Trips (in miles)')
plt.xticks (fontsize=14)
plt.ylabel('Number of Trips')
plt.yticks (fontsize=14)
plt.title('USA COVID TRAVEL BEHAVIOR-Week of September 15th,2020')
axes.title.set_size(25)
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
# PLOTTING STATE LEVEL DATA
#PLOTTING FOR CALIFORNIA
XCalifornia=A[(A['State Postal Code']=='CA') & (A['Date']=='20200706')& (A['Level']=='State') ].transpose()[9:]
XCalifornia.reset_index(inplace=True)
XCalifornia.columns=['Trip_Type',"Trip_Number"]
XCalifornia
#MATPLOTLIB DATA PLOTS CODE SPECIFICATIONS- STATE: CALIFORNIA
plt.figure(figsize=(20,10))
axes = plt.gca()
plt.xticks(rotation=45)
plt.bar(XCalifornia.Trip_Type,XCalifornia.Trip_Number,color='red')
plt.xlabel('Type of Trips (in miles)')
plt.xticks (fontsize=14)
plt.ylabel('Number of Trips')
plt.yticks (fontsize=14)
plt.title('CALIFORNIA TRAVEL BEHAVIOR-Week of July 6th, 2020')
axes.title.set_size(25)
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
#PLOTTING COUNTY LEVEL DATA
# Example: MIAMI DADE COUNTY, FLORIDA
Xcounty=A[(A['County Name']=='Miami-Dade County') & (A['Date']=='2020/07/06')].transpose()[9:]

Xcounty.reset_index(inplace=True)
Xcounty.columns=['Trip_Type',"Trip_Number"]
Xcounty
#MATPLOTLIB DATA PLOTS CODE SPECIFICATIONS- County: MIAMI DADE, FLORIDA
plt.figure(figsize=(20,10))
axes = plt.gca()
plt.xticks(rotation=45)
plt.bar(Xcounty.Trip_Type,Xcounty.Trip_Number,color='grey')
plt.xlabel('Type of Trips (in miles)')
plt.xticks (fontsize=14)
plt.ylabel('Number of Trips')
plt.yticks (fontsize=14)
plt.title('MIAMI DADE COUNTY, FL TRAVEL BEHAVIOR-Week of July 6th, 2020')
axes.title.set_size(25)
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
#TIME PLOT- CHECKING HISTORICAL TRENDS AND IMPACT OF COVID
S=A[(A['Level']=='State')&(A['State Postal Code']=='CA') & (A['Date']> '20180301') ]
S
# CALIFORNIA ANALYSIS- Population not Staying at Home
y=S['Population Not Staying at Home']
x=S['Date']

plt.figure(figsize=(20,10))
axes = plt.gca()
plt.xticks(rotation=45)
sns.lineplot(x,y,marker='o',color='red')
plt.title('CALIFORNIA STATE DATA-Impact of COVID')
plt.xticks (fontsize=14)
plt.yticks (fontsize=14)
axes.title.set_size(25)
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
# CALIFORNIA ANALYSIS- Population Staying at Home
y1=S['Population Staying at Home']
x=S['Date']

plt.figure(figsize=(20,10))
axes = plt.gca()
plt.xticks(rotation=45)
sns.lineplot(x,y1,marker='o',color='green')
plt.title('CALIFORNIA STATE DATA-Impact of COVID')
plt.xticks (fontsize=14)
plt.yticks (fontsize=14)
axes.title.set_size(25)
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
#TIME PLOT- CHECKING HISTORICAL TRENDS AND IMPACT OF COVID
N=A[(A['Level']=='National')&(A['Date']> '20180301')]
N
# USA NATIONAL ANALYSIS- Population Staying at Home
y=N['Population Not Staying at Home']
x=N['Date']

plt.figure(figsize=(20,10))
axes = plt.gca()
plt.xticks(rotation=45)
sns.lineplot(x,y,marker='o',color='green')
plt.title('USA National DATA-Impact of COVID')
plt.xticks (fontsize=14)
plt.yticks (fontsize=14)
axes.title.set_size(25)
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
# USA NATIONAL ANALYSIS- Population Staying at Home
y1=N['Population Staying at Home']
x=N['Date']

plt.figure(figsize=(20,10))
axes = plt.gca()
plt.xticks(rotation=45)
sns.lineplot(x,y1,marker='o',color='green')
plt.title('USA National DATA-Impact of COVID')
plt.xticks (fontsize=14)
plt.yticks (fontsize=14)
axes.title.set_size(25)
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
