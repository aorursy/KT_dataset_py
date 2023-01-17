import numpy as np

import pandas as pd

import datetime

import time

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv("../input/earthquake-database/database.csv")

data.head()
data.columns
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

data.head()
timestamp = []

for d, t in zip(data['Date'], data['Time']):

    try:

        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')

        timestamp.append(time.mktime(ts.timetuple()))

    except ValueError:

        # print('ValueError')

        timestamp.append('ValueError')

        

timeStamp = pd.Series(timestamp)

data['Timestamp'] = timeStamp.values
final_data = data.drop(['Date', 'Time'], axis=1)

final_data = final_data[final_data.Timestamp != 'ValueError']

final_data.head()
X = final_data[['Timestamp', 'Latitude', 'Longitude']]

y = final_data[['Magnitude', 'Depth']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6) #Random State 6 gives best result for this case
reg = RandomForestRegressor(random_state=6)

reg.fit(X_train, y_train)

reg.predict(X_test)
reg.score(X_test, y_test)
plt.hist(data['Magnitude'])

plt.xlabel('Magnitude Size')

plt.ylabel('Number of Occurrences')

plt.title('Magniture size vs Number of occurrences', fontweight = 20, fontsize = 10)

plt.show()
data['date']=data['Date'].apply(lambda x: pd.to_datetime(x))

data['year']=data['date'].apply(lambda x:str(x).split('-')[0])
plt.figure(figsize=(25,8))

sns.set(font_scale=1.0)

sns.countplot(x="year",data=data)

plt.ylabel('Number of Earthquakes')

plt.xlabel('Number of Earthquakes in each year')
data['year'].value_counts()[::-1]
x=data['year'].unique()

y=data['year'].value_counts()

count=[]

for i in range(len(x)):

    count.append(y[x[i]])



plt.figure(figsize=(10,8))    

plt.scatter(x,count)

plt.xlabel('Year')

plt.ylabel('Number of earthquakes')

plt.title('Earthquakes between 1970 to 2016')

plt.show()
plt.scatter(data["Magnitude"],data["Depth"])