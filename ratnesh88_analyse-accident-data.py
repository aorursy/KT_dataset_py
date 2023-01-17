import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

monthData = pd.read_csv("../input/Traffic accidents by month of occurrence 2001-2014.csv",sep=",")

timeData = pd.read_csv("../input/Traffic accidents by time of occurrence 2001-2014.csv")

monthData.info()
monthData.head()
monthData.groupby('STATE/UT').sum()['TOTAL'].plot.bar(figsize=(12,6),stacked=True,title="State wise accidents comparison")
monthData.groupby('YEAR').sum()['TOTAL'].plot(figsize=(12,6),title="Year wise accidents frequency")
monthData[monthData.columns[0:-1]].groupby('YEAR').sum().plot(figsize=(18,10),title="Year wise monthly accidents comparison")
monthData[monthData.columns[0:-1]].groupby('YEAR').sum().plot.bar(figsize=(18,10),title="Year wise monthly accidents comparison")
monthData.groupby('TYPE')['TOTAL'].sum().plot("bar",figsize=(12,6),title="Type wise accidents frequency")
monthData[['YEAR','TYPE','TOTAL']].head()
monthData[monthData.columns.drop(['TOTAL','YEAR'])].groupby('STATE/UT').sum().plot.bar(figsize=(13,10),stacked=True,title="State wise monthly accidents comparison")
monthData[monthData.columns.drop(['TOTAL','YEAR'])].groupby('TYPE').sum().plot.bar(figsize=(13,10),title="Accident types wise monthly accidents comparison")
timeData.info()
timeData[timeData.columns[0:-1]].groupby('YEAR').sum().plot(figsize=(12,8),title="Year wise timely accidents comparison")
timeData[timeData.columns.drop(['Total','YEAR'])].groupby('STATE/UT').sum().plot.bar(figsize=(13,10),stacked=True,title="State wise timely accidents comparison")
timeData[timeData.columns.drop(['Total','YEAR'])].groupby('TYPE').sum().plot.bar(figsize=(13,10),title="Accident types wise monthly accidents comparison")
monthData[monthData.columns[0:-1]][monthData['STATE/UT'] == 'Madhya Pradesh'].groupby('YEAR').sum().plot(figsize=(18,10),title="Year wise monthly accidents comparison of MP")
monthData[monthData['STATE/UT'] == 'Madhya Pradesh'][monthData.columns.drop(['TOTAL','YEAR'])].groupby('TYPE').sum().plot.bar(figsize=(13,10),title="Accident types wise monthly accidents comparison Madhya Pradesh")
monthData[monthData['STATE/UT'] == 'Uttar Pradesh'][monthData['TYPE'] != 'Road Accidents' ][monthData.columns.drop(['TOTAL','YEAR'])].groupby('TYPE').sum().plot.bar(figsize=(13,10),title="Monthly railway accidents comparison in Uttar Pradesh")
monthData[monthData['TYPE'] != 'Road Accidents'][monthData['STATE/UT'] == 'Uttar Pradesh'].groupby('YEAR').sum()['TOTAL'].plot(figsize=(12,6),title="Year wise Railway accidents in Uttar Pradesh")