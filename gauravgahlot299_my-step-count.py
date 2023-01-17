import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



steps = pd.read_csv("../input/AugustSteps.csv",parse_dates=True)
steps.head(20)
def replaceHours(hour):

    replaceString = ""

    if hour.endswith('am'):

        if hour.startswith('12'):

            replaceString += '00'

        elif hour.startswith('10') or hour.startswith('11'):

            replaceString += hour[:2]

        else:

            replaceString += '0'+hour[:1]

    if hour.endswith('pm'):

        if hour.startswith('12'):

            replaceString += hour[:2]

        else:

            replaceString += str(int(hour[:-2])+12)

    return(replaceString)

    

def changeFormat(slot):

    return replaceHours(slot[0])+'-'+replaceHours(slot[1])

    

steps['Hour Slot'] = steps['Hour Slot'].str.split('-').apply(changeFormat)  

steps.head()
steps.Date = steps.Date.str.split(', ')

steps['DayOfWeek'] = steps.Date.apply(lambda x:x[0])

steps['DateFormatted'] = steps.Date.apply(lambda x:x[1])

steps.drop('Date', inplace = True, axis = 1)

steps = steps[['DayOfWeek', 'DateFormatted', 'Hour Slot', 'Outdoor Steps', 'Basic Steps']]

steps.head()
df = steps[:]

df.fillna(0, inplace = True)

df['Total_Steps'] = df['Outdoor Steps'] + df['Basic Steps']

df_new = df.pivot_table(index = 'DateFormatted', columns = 'Hour Slot', values = 'Total_Steps', aggfunc = 'sum')

plt.figure(figsize = (15,10))

sns.heatmap(df_new)
plt.figure(figsize = (15,5))

sns.barplot(x = steps['DateFormatted'], y = steps['Outdoor Steps'])

plt.xticks(rotation = 90)
plt.figure(figsize = (15,5))

plt.subplots_adjust(wspace = 0.5)

plt.subplot(1,2,1)

sns.barplot(x = steps['DayOfWeek'], y = steps['Outdoor Steps'])

plt.xticks(rotation = 90)

plt.subplot(1,2,2)

sns.barplot(x = steps['DayOfWeek'], y = steps['Basic Steps'])

plt.xticks(rotation = 90)
df1 = steps.pivot_table(index='DateFormatted', values='Outdoor Steps', aggfunc = 'sum')

sns.distplot(df1, label = 'Outdoor Steps')

print("Mean:",df1.mean())

df2 = steps.pivot_table(index='DateFormatted', values='Basic Steps', aggfunc = 'sum')

sns.distplot(df2, label = 'Basic Steps')

print("Mean:",df2.mean())

plt.legend()
df3 = steps.pivot_table(index='Hour Slot', values='Outdoor Steps', aggfunc = 'mean')

df4 = steps.pivot_table(index='Hour Slot', values='Basic Steps', aggfunc = 'mean')

plt.figure(figsize = (15,5))

sns.distplot(df3, label = 'Outdoor Steps')

sns.distplot(df4, label = 'Basic Steps')

plt.legend()