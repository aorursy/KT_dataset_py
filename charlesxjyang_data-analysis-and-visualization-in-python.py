#Importing some packages

import numpy as np #linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualization and graphing



#importing data as Pandas Dataframe

df=pd.read_csv('../input/CA_2013_onwards.csv')



##Data Cleaning

#drops the following columns. axis=1 denotes that these are columns not rows

df = df.drop(['stop_time','fine_grained_location', 'police_department','driver_age','search_type'],axis=1)



#drops any rows that still have missing data

df = df.dropna()



#set index to the stop date column

df = df.set_index('stop_date')



#converts the index to Datetime object

df.index = pd.to_datetime(df.index)



df.head()



#Make a copy of df without changing original df

df_copy = df.copy()



#slice of all police stops from 2013-01-01 to 2013-01-31

df_copy[:'2013-01-31']



#all police stop from October to December 2013

df_copy['2013-10':'2013-12']



#Indexing with Boolean Masks: finding the first day of each month

df_copy[df_copy.index.day==1]



#Indexing with Boolean Masks: only returning 2013 data

df_copy[df_copy.index.year==2013]



#Indexing with Boolean Masks: only returning police stops made on Saturdays

df_copy[df_copy.index.weekday_name=='Saturday']



#Changing index to days of week

df_copy.index = df_copy.index.weekday_name



##Building Time Series Graphs



#number of arrests made in CA on each day

Time_series = df.groupby('stop_date').agg('count').iloc[:,0] 



#builds a date range starting at 2013-01-01

x = pd.date_range('2013-01-01',periods=len(Time_series)) 



plt.plot(x,Time_series) 

#Building Time Series Graph for 2013

plt.figure()

Time_series = df[:'2013'].groupby('stop_date').agg('count').iloc[:,0] 

x = pd.date_range('2013-01-01',periods=len(Time_series)) 

plt.plot(x,Time_series), plt.title('Traffic stops in 2013'), plt.ylabel('Traffic Stops per day')

plt.show()



#Building Time Series Graph for January 2013

plt.figure()

Time_series = df['2013-01'].groupby('stop_date').agg('count').iloc[:,0] 

x = pd.date_range('2013-01-01',periods=len(Time_series))

plt.plot(x,Time_series), plt.title('Traffic stops in Jan-2013'), plt.ylabel('Traffic Stops per day')

plt.show()



##Break it down into days of the week-January 2013

Jan = pd.DataFrame(df['2013-01'].groupby('stop_date').agg('count').iloc[:,0])

#convert date to name of weekday

Jan.index = Jan.index.weekday_name

Jan.columns = ['num_stops']

Jan.index.name = 'Weekday'

#rank by number of arrests on each day

Jan.sort_values(by='num_stops')

#total number of arrests in January by days of week

Jan.groupby('Weekday').agg('sum').sort_values(by='num_stops')

#bar plot of arrests by weekday

Jan.groupby('Weekday').agg('sum').sort_values(by='num_stops').plot.bar(), plt.title('Traffic Stops in California in Janaury 2013')
## Repeat the same time-series analysis for the entire dataset



#Count the total number of traffic stops made on each day

Weekday = pd.DataFrame(df.groupby('stop_date').agg('count').iloc[:,0])



#change the Weekday df index to the name of the weekday rather than the date

Weekday.index = Weekday.index.weekday_name



#Renaming columns and index

Weekday.columns = ['num_stops']

Weekday.index.name = 'Weekday'



#sort dataframe based on number of arrests on each day of the week

Weekday.sort_values(by='num_stops')



#total number of arrests in January by days of week

Weekday.groupby('Weekday').agg('sum').sort_values(by='num_stops')



#bar plot of arrests by weekday

Weekday.groupby('Weekday').agg('sum').sort_values(by='num_stops').plot.bar(), plt.title('Traffic Stops in CA 2013 onwards')
##Analyzing Search Rate and Hit Rate by Race for California

#creating a dataframe with driver race as index

Data_race = df.groupby('driver_race').agg({'state':'count','search_conducted':'sum','contraband_found':'sum'})

Data_race.columns = ['stop_count','search_conducted','contraband_found']

Data_race['search_rate'] = (Data_race.loc[:,'search_conducted']/Data_race.loc[:,'stop_count'])*100

Data_race['hit_rate'] = (Data_race.loc[:,'contraband_found']/Data_race.loc[:,'search_conducted'])*100



Data_race.iloc[:,[0,3,4]]



#Plotting Search Rate and Hit Rate by Race for California

Data_race.iloc[:,[3]].plot.bar(), plt.ylabel('Search Rate(%)'), plt.xlabel('Driver Race'), plt.title('Search Rate in California by Driver Race')

Data_race.iloc[:,[4]].plot.bar(), plt.ylabel('Hit Rate(%)'), plt.xlabel('Driver Race'), plt.title('Hit Rate in California by Driver Race')
#North Carolina Data

d = {'search_rate': [3.1, 5.4, 4.1, 1.7],'hit_rate': [32, 29, 19, 36]}

NC = pd.DataFrame(d,index=['White', 'Black', 'Hispanic', 'Asian'],columns=['search_rate','hit_rate'])



#Plotting Search Rate and Hit Rate by Race for North Carolina

NC.loc[:,'search_rate'].plot.bar(), plt.ylabel('Search Rate(%)'), plt.xlabel('Driver Race')

plt.title('Search Rate in North Carolina by Driver Race'), plt.show()

NC.loc[:,'hit_rate'].plot.bar(), plt.ylabel('Hit Rate(%)'), plt.xlabel('Driver Race')

plt.title('Hit Rate in North Carolina by Driver Race'), plt.show()

NC
#Plotting Search Rates of CA and NC

plt.figure()



#creating positions of each bar

pos = list(range(len(NC)))



#set width of each bar

width = 0.25

fig, ax = plt.subplots(figsize=(10,5))



#Create bars for California Data

plt.bar(np.array(pos), #position of bars

        Data_race.loc[['Asian','Black','Hispanic','White'],'search_rate'], #Removed "Others" b/c not included in NC data

        # of width

        width,

        # color of CA bars

        color='blue')

#Create bars for North Carolina Data

plt.bar(np.array([p + width for p in pos]), #position of bars shifted over to avoid overlap

        NC['search_rate'],

        # of width

        width,

        # color of NC bars

        color='red')



plt.ylabel('Search Rate(%)')

plt.xlabel('Driver Race')

plt.legend(['California','North Carolina'])

plt.title('Comparing Search Rate in North Carolina and California by Driver Race')

plt.xticks([0,1,2,3,4],['Asian','Black','Hispanic','White'])

plt.show()
#Plotting Hit Rates of CA and NC

plt.figure()



#creating positions of each bar

pos = list(range(len(NC)))



#set width of each bar

width = 0.25

fig, ax = plt.subplots(figsize=(10,5))



#Create bars for California Data

plt.bar(np.array(pos), #position of bars

        Data_race.loc[['Asian','Black','Hispanic','White'],'hit_rate'], #Removed "Others" b/c not included in NC data

        # of width

        width,

        # color of CA bars

        color='blue')

#Create bars for North Carolina Data

plt.bar(np.array([p + width for p in pos]), #position of bars shifted over to avoid overlap

        NC['hit_rate'],

        # of width

        width,

        # color of NC bars

        color='red')



plt.ylabel('Hit Rate(%)')

plt.xlabel('Driver Race')

plt.legend(['California','North Carolina'])

plt.xticks([0,1,2,3,4],['Asian','Black','Hispanic','White'])

plt.title('Comparing Hit Rate in North Carolina and California by Driver Race')

plt.show()
#Analyzing Search Rate and Hit Rate by Race and County for California

Data = df.groupby(['county_name','driver_race']).agg({'state':'count','search_conducted':'sum','contraband_found':'sum'})



Data.columns = ['stop_count','search_conducted','contraband_found']



Data['search_rate'] = (Data.loc[:,'search_conducted']/Data.loc[:,'stop_count'])*100



Data['hit_rate'] = (Data.loc[:,'contraband_found']/Data.loc[:,'search_conducted'])*100



#Removing smaller counties with less than 5 successful contraband hits

for county, row, in Data.groupby(level=0):

    if ((row.loc[:,'contraband_found']<5).any())==True:

        Data.drop(county,inplace=True)

    elif (row.xs(county,level=0).loc[:,'contraband_found'].empty)==True:

        continue

    else:

        continue

        

Data.iloc[:,[0,3,4]]
##Graphing Search rates

#Blacks

plt.figure()

plt.scatter(Data.xs('White',level=1).loc[:,'search_rate'],

            Data.xs('Black',level=1).loc[:,'search_rate'])

plt.ylim(0,2),plt.xlim(0,2),plt.plot([0,3],[0,3]),

plt.xlabel('White Search Rate(%)'),plt.ylabel('Black Search Rate(%)'),plt.plot()



#Hispanics

plt.figure()

plt.scatter(Data.xs('White',level=1).loc[:,'search_rate'],

            Data.xs('Hispanic',level=1).loc[:,'search_rate'])

plt.ylim(0,3), plt.xlim(0,3), plt.plot([0,3],[0,3]),

plt.xlabel('White Search Rate(%)'),plt.ylabel('Hispanic Search Rate(%)'),plt.plot()



#Asians

plt.figure()

plt.scatter(Data.xs('White',level=1).loc[:,'search_rate'],

            Data.xs('Asian',level=1).loc[:,'search_rate'])

plt.ylim(0,2), plt.xlim(0,2), plt.plot([0,3],[0,3]),

plt.xlabel('White Search Rate(%)'),plt.ylabel('Asian Search Rate(%)'),plt.plot()

#Identifying Outlier County

#We can use idxmax for search rates of blacks because in every case,

#the county that is farthest away from the equality line also has the highest search rate of blacks

Outlier_black_search = [Data.xs('Black',level=1)['search_rate'].idxmax()]

Outlier_hispanic_search = [Data.xs('Hispanic',level=1)['search_rate'].idxmax()]

print(Outlier_black_search,Outlier_hispanic_search)

##Hit rates

#White-Black

plt.figure()

plt.scatter(Data.xs('White',level=1).loc[:,'hit_rate'],

            Data.xs('Black',level=1).loc[:,'hit_rate']),

#highlighting Trinity County on the Scatterplot

plt.scatter(Data.loc[('Trinity County','White'),'hit_rate'],

            Data.loc[('Trinity County','Black'),'hit_rate'],color='red')

plt.xlim(0,100),plt.ylim(0,100), plt.plot([0,100],[0,100]), plt.xlabel('White Search Rate'), plt.ylabel('Black Search Rate')



#White-Hispanics

plt.figure()

plt.scatter(Data.xs('White',level=1).loc[:,'hit_rate'],

            Data.xs('Hispanic',level=1).loc[:,'hit_rate'])

#highlighting Trinity County on the Scatterplot

plt.scatter(Data.loc[('Trinity County','White'),'hit_rate'],

            Data.loc[('Trinity County','Hispanic'),'hit_rate'],color='red')

plt.ylim(0,100), plt.xlim(0,100), plt.plot([0,100],[0,100]), plt.xlabel('White Search Rate'), plt.ylabel('Hispanic Search Rate')



#White-Asians

plt.figure()

plt.scatter(Data.xs('White',level=1).loc[:,'hit_rate'],

            Data.xs('Asian',level=1).loc[:,'hit_rate'])

#highlighting Trinity County on the Scatterplot

plt.scatter(Data.loc[('Trinity County','White'),'hit_rate'],

            Data.loc[('Trinity County','Asian'),'hit_rate'],color='red')

plt.ylim(0,100), plt.xlim(0,100), plt.plot([0,100],[0,100]), plt.xlabel('White Search Rate'), plt.ylabel('Asian Search Rate')