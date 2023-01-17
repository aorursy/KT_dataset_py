"""
Author: Carlos Abiera
Date: 9 October 2018

"""
import pandas as pd
import re
from matplotlib import pyplot as plt

#READ AND CULLOUT COLUMN
data = pd.read_csv("../input/tel.csv") #read file

# display top rows using head 
data.head(10)
# print summary statistics
data.describe()
# display all the column names in the data
data.columns
data['Status'].value_counts()
# plot the value counts of sex 
data['Status'].value_counts().plot.bar()
# print the top 10 ages
data['Status'].value_counts().head(10)
#remove the less important columns
data.drop(['AccountCode', 'Ring Group','SrcChannel', 'DstChannel'], axis=1, inplace=True) #drop columns 

#show the essential columns and the top 5 rows
data.head()
#CLEAN DESTINATION REMOVE FMGR
data['Destination'] = data['Destination'].str.extract('(\d+)', expand=False)
# check if the function above is successful
data.head()
data['type'] = pd.np.where(data.Destination.str.len() > 9, '2',
               pd.np.where(data.Destination.str.len()>5, '1',  '0'))    

# check if the function above is successful
data.head()
#LANDLINE VS MOBILE VS LOCAL

nsti_usage = data.groupby('type')['ID'].count()
u = pd.DataFrame(nsti_usage)
u['Label'] = ['Local','Landline','Mobile']

labels = u['Label'].tolist()
sizes = u['ID'].tolist()
explode = (0, 0.1,  0)  # only "explode" the 2nd slice (i.e. 'Hogs')


fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
duration = data['Duration'].str.replace(r"\(.*\)","") #remove all char including parethesis in duration
duration = duration.str.replace("s","") #remove all seconds (s) character
data['Duration'] = pd.to_numeric(duration) #convert series to numeric
# check if the function above is successful
data.head()
#CLEAN DATE ADD MONTH COLUMN
d = pd.to_datetime(data['Date']) #convert column to datetime data type
data['ByDate'] = d.dt.date #assign converted value to row
data['ByMonth'] = d.dt.month  #Add month
data['ByDayWeek'] = d.dt.dayofweek  #Add hour
# check if the function above is successful
data.head()
#MONTHLY USAGE
dm = data.groupby('ByMonth', axis=0)['ID'].count() #good but try to check #2018-01-23    844
dm = pd.DataFrame(dm)
# u['Label'] = ['Local','Landline','Mobile']
month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September']
dm['MONTH'] = month
dm_label = dm['MONTH'].tolist()
dm_size = dm['ID'].tolist()
dm.plot.bar()

#You might wonder that the bar graph below only shows the 9th month because I only got until september



#DAILY USAGE
dy = data.groupby('ByDayWeek', axis=0)['ID'].count() #good but try to check #2018-01-23    844
dy = pd.DataFrame(dy)
day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dy['Day'] = day
dy_label = dy['Day'].tolist()
dy_size = dy['ID'].tolist()
dy.plot.bar()

# 0 as monday
# QUERY SOURCE, STATUS, AND MONTH
office = data.loc[(data['Source'] == 1000) & (data['ByMonth'] == 1)  ]
#SHOW TOP 5 CALLS FROM 1000
office.head()
o = data.loc[data['type'] == '2']
#SHOW TOP 5 LONG DISTANCE CALLS 
o.head()








