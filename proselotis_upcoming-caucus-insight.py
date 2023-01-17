import numpy as np 

import pandas as pd 



data = pd.read_csv("/kaggle/input/2020-democratic-primary-endorsements/endorsements-2020.csv")

data['date'] = pd.to_datetime(data['date'])
import matplotlib.pyplot as plt

%matplotlib inline

names, values = data['endorsee'].value_counts().index, data['endorsee'].value_counts().values

plt.bar(names,values)

plt.xticks(rotation = 90)

plt.title("Total Endorsments")

plt.xlabel("Candidate name")

plt.ylabel("Number of Endorsments")
data2019 = data[data['date'] > pd.datetime(2019,1,1)]



names, values = data2019['endorsee'].value_counts().index, data2019['endorsee'].value_counts().values

plt.bar(names,values)

plt.xticks(rotation = 90)

plt.title("Total Endorsments since 2019")

plt.xlabel("Candidate name")

plt.ylabel("Number of Endorsments")
data2020 = data[data['date'] > pd.datetime(2020,1,1)]



names, values = data2020['endorsee'].value_counts().index, data2020['endorsee'].value_counts().values

plt.bar(names,values)

plt.xticks(rotation = 90)

plt.title("Total Endorsments in 2020")

plt.xlabel("Candidate name")

plt.ylabel("Number of Endorsments")
iaData = data[data['state'] == 'IA']

names, values = iaData['endorsee'].value_counts().index, iaData['endorsee'].value_counts().values

plt.bar(names,values)

plt.xticks(rotation = 90)

plt.title("Total Endorsments from Iowa")

plt.xlabel("Candidate name")

plt.ylabel("Number of Endorsments")
uniqueStatesNum = len(np.unique(data['state']))

bernieAveragePerState = len(data[data['endorsee'] == 'Bernie Sanders']) / uniqueStatesNum

peteAveragePerState = len(data[data['endorsee'] == 'Pete Buttigieg']) / uniqueStatesNum

print("Bernie average endorsment per state: ",bernieAveragePerState)

print("Pete   average endorsment per state: ",peteAveragePerState)

print("Bernie endorsments in Iowa         : ",len(iaData[iaData['endorsee'] == 'Bernie Sanders']))

print("Pete   endorsments in Iowa         : ",len(iaData[iaData['endorsee'] == 'Pete Buttigieg']))
neighborStatesIA = data[data['state'].isin(['IL','WI','MN','SD','NE','MO'])]

names, values = neighborStatesIA['endorsee'].value_counts().index, neighborStatesIA['endorsee'].value_counts().values

plt.bar(names,values)

plt.xticks(rotation = 90)

plt.title("Total Endorsments From IA Neighbors")

plt.xlabel("Candidate name")

plt.ylabel("Number of Endorsments")
nhData = data[data['state'] == 'NH']

names, values = nhData['endorsee'].value_counts().index, nhData['endorsee'].value_counts().values

plt.bar(names,values)

plt.xticks(rotation = 90)

plt.title("Total Endorsments from New Hampshire")

plt.xlabel("Candidate name")

plt.ylabel("Number of Endorsments")
neighborStatesNH = data[data['state'].isin(['MA','VT','MA'])]

names, values = neighborStatesNH['endorsee'].value_counts().index, neighborStatesNH['endorsee'].value_counts().values

plt.bar(names,values)

plt.xticks(rotation = 90)

plt.title("Total Endorsments From NH Neighbors")

plt.xlabel("Candidate name")

plt.ylabel("Number of Endorsments")
# Average points for each canidate by endorser 

data.groupby('endorsee')['points'].mean().sort_values(ascending = False)
for cat in np.unique(data['category']):

    catSpecific = data[data['category'] == cat]

    print(cat + ":",'\n')

    print(catSpecific.groupby('endorsee')['date'].count().sort_values(ascending = False).to_string(),"\n")