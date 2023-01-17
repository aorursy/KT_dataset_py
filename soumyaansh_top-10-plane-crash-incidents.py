import matplotlib.pyplot as plt  # Data visualization

import numpy as np

import pandas as pd  # CSV file I/O, e.g. the read_csv function

import seaborn as sns

from collections import Counter

from tabulate import tabulate

import calendar



# load the data and create dataframe

crash_data = pd.read_csv("../input/3-Airplane_Crashes_Since_1908.txt")



print("\nColumns and their datatypes in dataframe :\n")

print(crash_data.dtypes)



# get rid of first part of the Location value, keep the larger entity of the area(country or famous city)

def processLocation(df):

    df.Location = df.Location.apply(lambda x: x.split(",")[-1])

    return df





# Location,Date and Operator are the basic and very important features

# To perform any analysis we need these features so dropping all the empty or null values

def removeMissindFields(df):

    df = df.dropna(subset=['Location', 'Date', 'Operator'])

    return df





def removeFeatures(df):

    df = df.drop(['Route', 'cn/In', 'Registration', 'Summary', 'Ground', 'Flight #'], axis=1)

    return df





crash_data = removeMissindFields(crash_data)

crash_data = removeFeatures(crash_data)

crash_data = processLocation(crash_data)

print("\nDeadly month analysis with maximum crashes \n")



# convert the Date datatype from object to datetime to fetch Month and year 

crash_data['Date'] = pd.to_datetime(crash_data['Date'])



crash_data['Year'], crash_data['Month'] = crash_data['Date'].dt.year, crash_data['Date'].dt.month

crash_data['MonthName'] = crash_data['Month'].apply(lambda x: calendar.month_name[x])



# count the number of crashes occured over the period of time

crash_month = crash_data.MonthName.value_counts()



# change the seried to dataframe for easier plotting of graphs

crash_month_df = crash_month.to_frame().reset_index()

crash_month_df.columns = ['Month', 'Crashes']



#change the datatype from object to float for graph plotting

crash_month_df[['Crashes']] = crash_month_df[['Crashes']].astype(float)

print(crash_month_df)



#plot the graph

sns.barplot(crash_month_df["Month"], crash_month_df["Crashes"], palette="Set3")

plt.xticks(rotation=90)

plt.xlabel('Month')

plt.ylabel('Number of Crashes')

plt.show()



print("\nDeadly year analysis with maximum crashes \n")



# count the number of crashes occured over the period of time

crash_year = crash_data.Year.value_counts()



# change the seried to dataframe for easier plotting of graphs

crash_year_df = crash_year.to_frame().reset_index()

crash_year_df.columns = ['Year', 'Crashes']



#change the datatype from object to float for graph plotting

crash_year_df[['Crashes']] = crash_year_df[['Crashes']].astype(float)

crash_year_df_10 = crash_year_df.head(10)

print(crash_year_df_10)





#plot the graph

sns.barplot(crash_year_df_10["Year"], crash_year_df_10["Crashes"], palette="BuGn_d")

plt.xticks(rotation=90)

plt.xlabel('Year')

plt.ylabel('Number of Crashes')

plt.figure(figsize=(20,10))

plt.show()


print("\nTop 10 Locations with maximum crashes \n")

loc_list = Counter(crash_data['Location']).most_common(10)

print(tabulate(loc_list, headers=['Location', 'Number of crashes']))



print("\nTOP 30 locations with maximum crashes \n")

# count the countries with number of creashes using value_counts() function

loc_count = crash_data['Location'].value_counts()



# change the seried to dataframe for easier plotting of graphs

loc_count_df = loc_count.to_frame().reset_index()

loc_count_df.columns = ['Location', 'Crashes']



#change the datatype from object to float for graph plotting

loc_count_df[['Crashes']] = loc_count_df[['Crashes']].astype(float)



# save only top 10 values from dataframe

loc_count_df_30 = loc_count_df.head(30)  #get the 30 most dangeroud countries to fly



#plot the graph

sns.barplot(loc_count_df_30["Location"], loc_count_df_30["Crashes"], palette="BuGn_d")

plt.xticks(rotation=90)

plt.xlabel('Locations')

plt.ylabel('Number of Crashes')

plt.show()

print("\nTop 30 Locations with maximum crashes \n")

sns.swarmplot(x="Crashes", y="Location",  data=loc_count_df_30);

plt.xticks(rotation=90)

plt.show()
print("\nTop 10 operator with maximum crashes \n")

oper_list = Counter(crash_data['Operator']).most_common(10)

print(tabulate(oper_list, headers=['Operator Name', 'Number of crashes']))





print("\nTOP 10 locations with maximum crashes \n")

# count the countries with number of creashes using value_counts() function

oper_count = crash_data['Operator'].value_counts()



# change the seried to dataframe for easier plotting of graphs

oper_df = oper_count.to_frame().reset_index()

oper_df.columns = ['Operator', 'Crashes']



#change the datatype from object to float for graph plotting

oper_df[['Crashes']] = oper_df[['Crashes']].astype(float)



# save only top 10 values from dataframe

oper_df_10 = oper_df.head(10)  #get the 10 most dangeroud countries to fly



#plot the graph

sns.barplot(oper_df_10["Operator"], oper_df_10["Crashes"], palette="colorblind")

plt.xticks(rotation=90)

plt.xlabel('Operator')

plt.ylabel('Number of Crashes')

plt.show()
crash_data =  crash_data.dropna(subset=['Time'])



print("\nTop 10 Time with maximum crashes \n")

time_list = Counter(crash_data['Time']).most_common(10)

print(tabulate(time_list, headers=['Time', 'Number of crashes']))



crash_time_count = crash_data['Time'].value_counts()

# change the seried to dataframe for easier plotting of graphs

crash_time_count_df = crash_time_count.to_frame().reset_index()

crash_time_count_df.columns = ['Time', 'Crashes']



#change the datatype from object to float for graph plotting

crash_time_count_df[['Crashes']] = crash_time_count_df[['Crashes']].astype(float)



# save only top 10 values from dataframe

crash_time_count_df_10 = crash_time_count_df.head(10)



sns.barplot(crash_time_count_df_10["Time"], crash_time_count_df_10["Crashes"], palette="Set3")

plt.xticks(rotation=90)

plt.xlabel('Time')

plt.ylabel('Number of Crashes')

plt.show()