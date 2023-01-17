#libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/uber-request-data/Uber Request Data.csv")
df.head()
print(df.shape)
print(df.info())
#Check for duplicates in Request id
ids = df[['Request id']]
is_duplicate = ids.duplicated()
sum(is_duplicate)
#Check for any missing values in the data set
df.isnull().sum()
cars_NA = len(df[df['Status'] == 'No Cars Available'])
trip_cancel = len(df[df['Status'] == 'Cancelled'])
print("No. of request not accepted due to unavailability of cars: " + str(cars_NA))
print("No. of unattended requests (cars not available + driver cancelled): " + str(cars_NA+trip_cancel))
#Correcting the data types
df['Request timestamp'] = pd.to_datetime(df['Request timestamp'])
df['Drop timestamp'] = pd.to_datetime(df['Drop timestamp'])

df.head()
df.info()
#Extracting pick-up and drop time, day, hour for further analysis
df['pick_date'] = df['Request timestamp'].dt.date
df['pick_day'] = df['Request timestamp'].dt.day
df['pick_hour'] = df['Request timestamp'].dt.hour


df['drop_date'] = df['Drop timestamp'].dt.date
df['drop_day'] = df['Drop timestamp'].dt.day
df['drop_hour'] = df['Drop timestamp'].dt.hour

df.head()
#Cross checking
print(df['pick_date'].unique())
print("\n")
print(df['drop_date'].unique())
print(df['drop_date'].value_counts())
#Plot a countplot on "Status" column to identify count of completed trips, cancelled, and no cars available. 
plt.figure(figsize=(8, 5))
g = sns.countplot(x="Status", data=df)
sns.despine()
plt.title("Frequency of request", fontsize = 18)
plt.xlabel("Trip status", fontsize = 14)
plt.ylabel("No. of trips", fontsize = 14)

for p in g.patches:
    g.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height() + 50), ha = 'center', va = 'center')

plt.show()
grp = df.groupby('Status')['Request id'].count()

percent_completed = grp["Trip Completed"]/ len(df) * 100
percent_cancel = grp['Cancelled']/ len(df) * 100
percent_nocars = grp['No Cars Available']/ len(df) * 100

print("The percentage of trips completed: " + "{:.2f}".format(percent_completed))
print("The percentage of trips cancelled: " + "{:.2f}".format(percent_cancel))
print("The percentage of requests cancelled due to unavailability of cabs: " + "{:.2f}".format(percent_nocars))
#Plot a countplot on a "Pick point" column to identify count of requests from Airport and City

plt.figure(figsize=(5, 4))
l = sns.countplot(x = "Pickup point", data = df)
sns.despine()
plt.title("Frequency of cab requests", fontsize = 18)
plt.xlabel("Pick Up Point", fontsize = 14)
plt.ylabel("Count of Cab request", fontsize = 14)

for p in l.patches:
    l.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height() + 70), ha = 'center', va = 'center')

plt.show()
#cab requests w.r.t. location
plt.figure(figsize=(8, 5))
l = sns.countplot(x = "Pickup point", hue = "Status", data = df)
sns.despine()
plt.title("Cab Requests Serviceability wrt Location", fontsize = 18)
plt.xlabel("Pick Up Point", fontsize = 14)
plt.ylabel("Count of Cab Requests", fontsize = 14)

for p in l.patches:
    l.annotate(p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height() + 30), ha = 'center', va = 'center')

plt.show()
#Plotting cab sequests by day
plt.figure(figsize=(8,4))
sns.countplot(x='pick_date', data=df)
sns.despine()
plt.title("Requests by Day", fontsize = 18)
plt.xlabel("Day", fontsize = 14)
plt.ylabel("Request Count", fontsize = 14)
plt.show()
#Plot count plot for all days w.r.t. to pick up hour
plt.figure(figsize=(10, 4))
sns.countplot(x = "pick_hour", hue = "Pickup point", data = df)
sns.despine()
plt.title("Distribution of cab requests on hourly basis", fontsize = 18)
plt.xlabel("Pick-up Hour", fontsize = 14)
plt.ylabel("Count of Cab Requests", fontsize = 14)
plt.show()
# function to create a time slot for various time period of day
def time_period(x):
    'divide the time of the day into four categories'
    if 2<= x < 5:
        return "Pre_Morning"
    elif 5 <= x < 10:
        return "Morning_Rush"
    elif 10 <= x < 17:
        return "Day_Time"
    elif 17 <= x < 22:
        return "Evening_Rush"
    else:
        return "Late_Night"

df['time_slot'] = df.pick_hour.apply(lambda x: time_period(x))

df.time_slot.value_counts()
#Plot requests w.r.t. status for different time slots
plt.figure(figsize=(10, 5))
sns.countplot(x = "time_slot", data = df, 
              order= ['Pre_Morning', 'Morning_Rush', "Day_Time", "Evening_Rush", "Late_Night"])
sns.despine()
plt.title("Number of cab requests in different time slots", fontsize = 18)
plt.xlabel("Time Slots", fontsize = 14)
plt.ylabel("Count of Cab Requests Serviced", fontsize = 14)
plt.show()
#Plot requests w.r.t. status for different time slots at airport/city
plt.figure(figsize=(10, 5))
sns.countplot(x = "time_slot", data = df, hue = 'Pickup point',
              order= ['Pre_Morning', 'Morning_Rush', "Day_Time", "Evening_Rush", "Late_Night"])
sns.despine()
plt.title("Number of cab requests at different time at airport/ city", fontsize = 18)
plt.xlabel("Time Slots", fontsize = 14)
plt.ylabel("Count of Cab Requests", fontsize = 14)
plt.show()
#Plot requests w.r.t. status for different time slots
plt.figure(figsize=(10, 5))
sns.countplot(x = "time_slot", hue = "Status", data = df, 
              order= ['Pre_Morning', 'Morning_Rush', "Day_Time", "Evening_Rush", "Late_Night"])
sns.despine()
plt.title("Cab requests serviceability in different time slots", fontsize = 18)
plt.xlabel("Time Slots", fontsize = 14)
plt.ylabel("Count of Cab Requests Serviced", fontsize = 14)
plt.show()
airport = df[df['Pickup point'] == 'Airport']
city = df[df['Pickup point'] == 'City']
#Plot requests w.r.t. status for different time slots at the airport
plt.figure(figsize=(10, 5))
sns.countplot(x = "time_slot", hue = "Status", data = airport,
              order= ['Pre_Morning', 'Morning_Rush', "Day_Time", "Evening_Rush", "Late_Night"])
sns.despine()
plt.title("Cab requests serviceability at different time slots at the Airport", fontsize = 18)
plt.xlabel("Time Slots", fontsize = 14)
plt.ylabel("Count of Cab Requests Serviced", fontsize = 14)
plt.show()
#Plot requests w.r.t. status for different time slots in the city
plt.figure(figsize=(10, 5))
sns.countplot(x = "time_slot", hue = "Status", data = city,
              order= ['Pre_Morning', 'Morning_Rush', "Day_Time", "Evening_Rush", "Late_Night"])
sns.despine()
plt.title("Cab requests serviceability at different time slots in the City", fontsize = 18)
plt.xlabel("Time Slots", fontsize = 14)
plt.ylabel("Count of Cab Requests Serviced", fontsize = 14)
plt.show()
airport_analysis = pd.DataFrame(airport.groupby('pick_hour')['Status'].count())
airport_analysis = airport_analysis.rename(columns = {'Status' : 'Demand'})
airport_analysis.head(3)
airport_completion = airport[airport['Status'] == 'Trip Completed']

airport_analysis['Supply'] = pd.DataFrame(airport_completion.groupby('pick_hour')['Status'].count())
airport_analysis.head(3)
airport_analysis['Gap'] = airport_analysis['Demand'] - airport_analysis['Supply']
plt.figure(figsize=(8, 4))
airport_analysis.plot(kind = 'line')
plt.title("Supply-Demand Gap at the Airport", fontsize = 18)
plt.xlabel("Pick-up Hours", fontsize = 14)
plt.ylabel("Count", fontsize = 14)
plt.show()
city_analysis = pd.DataFrame(city.groupby('pick_hour')['Status'].count())
city_analysis = city_analysis.rename(columns = {'Status' : 'Demand'})
city_analysis.head(3)
city_completion = city[city['Status'] == 'Trip Completed']

city_analysis['Supply'] = pd.DataFrame(city_completion.groupby('pick_hour')['Status'].count())
city_analysis.head(3)
city_analysis['Gap'] = city_analysis['Demand'] - city_analysis['Supply']
plt.figure(figsize=(8, 4))
city_analysis.plot(kind = 'line')
plt.title("Supply-Demand Gap in the City", fontsize = 18)
plt.xlabel("Pick-up Hours", fontsize = 14)
plt.ylabel("Count", fontsize = 14)
plt.show()
#Let's create pie charts instead of a count plots
def pie_chart(dataframe):
    """
    creates a pie chart
    input: dataframe with a 'category' as index and a numerical column
    output: pie chart
    """
    labels = dataframe.index.values
    sizes = dataframe['Status'].values
        
    fig1, ax1 = plt.subplots()
    fig1 = plt.figure(figsize=(8, 4))
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
#Status of trips at Airport in the evening rush time
df_airport = airport[airport.time_slot == "Evening_Rush"]
df_airport_count = pd.DataFrame(df_airport.Status.value_counts())
pie_chart(df_airport_count)
#Status of trips in the city in the morning rush time 
df_city = city[city.time_slot == "Morning_Rush"]
df_city_count = pd.DataFrame(df_city.Status.value_counts())
pie_chart(df_city_count)