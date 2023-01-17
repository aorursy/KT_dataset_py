import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
uber_data = pd.read_csv('../input/ubercsv/Uber Request Data.csv')
print(uber_data.shape)
print(uber_data.columns)
uber_data.head()
uber_data.info()
#Converting Request_timestamp and drop_timestamp to uniform datetime format

uber_data["Request timestamp"] = uber_data["Request timestamp"].apply(lambda x: pd.to_datetime(x))

uber_data["Drop timestamp"] = uber_data["Drop timestamp"].apply(lambda x: pd.to_datetime(x))

uber_data.head()
#Check for null values
uber_data.isnull().sum()
uber_data.Status.value_counts()
#Extract the hour from requested timestamp
uber_data["Request hour"] = uber_data["Request timestamp"].dt.hour
uber_data.head(5)
plt.hist(uber_data["Request hour"],edgecolor='black',bins=24)
plt.xlabel("Request hour")
plt.ylabel("No. of Requests")
plt.show()
#Divide the time of the day into five categories
def time_period(x):
    if x < 5:
        return "Early Morning"
    elif 5 <= x < 10:
        return "Morning"
    elif 10 <= x < 17:
        return "Day Time"
    elif 17 <= x < 22:
        return "Evening"
    else:
        return "Late Night"
uber_data['Time slot'] = uber_data['Request hour'].apply(lambda x: time_period(x))
uber_data.head()
uber_data['Time slot'].value_counts().plot.bar()
plt.show()
uber_data["Pickup point"].value_counts().plot.pie(autopct='%1.0f%%')
plt.show()
uber_data["Status"].value_counts().plot.pie(autopct='%1.0f%%')
plt.show()
plt.style.use('ggplot')
uber_data.groupby(['Time slot','Status']).Status.count().unstack().plot.bar(legend=True, figsize=(15,10))
plt.title('Total Count of all Trip Statuses')
plt.xlabel('Sessions')
plt.ylabel('Total Count of Trip Status')
plt.show()
