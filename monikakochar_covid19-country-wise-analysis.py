# Call libraries
%reset -f
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
%matplotlib inline

# Display multiple outputs from a jupyter cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#os.chdir("D:\\Data")
# DateFrame object is created while reading file available at particular location given below

df = pd.read_csv("../input/uncover/WHO/world-health-organization-who-situation-reports.csv",parse_dates = ['date'])
# Displaying First 10 Records of DataFrame

df.head(10)
# Grouping the data with country column

gr1 = df.groupby(['location'])

# Calculating the country-wise sum of new cases, new deaths and total cases and displaying 10 records

gr1.agg({'new_cases' : np.sum,'new_deaths': np.sum, 'total_cases' : np.sum}).head(10)
# Create a subset of DataFrame of countries China and Italy

df1 = df[(df['location']=='Italy') | (df['location']=='China')]

columns = ['total_cases', 'new_cases', 'new_deaths']
           
# Create a Figure Object
           
fig = plt.figure(figsize = (10,5))

# Using for loop to plot all at once

for i in range(len(columns)):
    plt.subplot(1,3,i+1)
    sns.distplot(df1[columns[i]])
    sns.despine()
Columns= ['new_cases','new_deaths','total_cases']
Cat_Var= ['location']

# First create pairs of cont and cat variables

mylist = [(cont,cat)  for cont in Columns  for cat in Cat_Var]

#mylist

# Create a Figure Object

fig = plt.figure(figsize = (5,5))

# Using for loop to plot all boxplot at once

for i, k in enumerate(mylist):
    plt.subplot(3,1,i+1)
    sns.boxplot(x = k[1], y = k[0], data = df1)
Columns= ['new_cases','new_deaths','total_cases']
Cat_Var= ['location']

# First create pairs of cont and cat variables

mylist = [(cont,cat)  for cont in Columns  for cat in Cat_Var]

#mylist

# Create a Figure Object

fig = plt.figure(figsize = (5,5))

# Using for loop to plot boxplot again to see the impact of notch 

for i, k in enumerate(mylist):
    plt.subplot(3,1,i+1)
    sns.boxplot(x = k[1], y = k[0], data = df1, notch = True)
# Create a Figure Object

fig = plt.figure(figsize = (20,5))

# Using loop to plot all bar plot at once

for i, k in enumerate(mylist):   
     plt.subplot(1,3,i+1)
     sns.barplot(x = k[1], y = k[0], data = df1, color ='b')
# Create a Figure Object

fig = plt.figure(figsize = (5,5))

# Displaying relationship of new cases and new deaths by kde(or countour) plot

sns.jointplot(df1.new_cases,df1.new_deaths,kind='kde',cmap =plt.cm.coolwarm)
# Create a Figure Object

fig = plt.figure(figsize = (5,5))

# Displaying relationship of new cases and total_cases by hex plot

sns.jointplot(df1.new_cases,df1.total_cases,kind="hex",cmap =plt.cm.coolwarm)
# Create a Figure Object

fig = plt.figure(figsize = (5,5))

# Displaying relationship of new cases and total_cases by reg plot

sns.jointplot(df1.new_cases,df1.total_cases,kind="reg")
# A shallow copy of DataFrame is created

df2 = df1.copy()

# New column is created by extracting only month from date

df2['month'] = df2['date'].dt.month

# DateFrame is grouped by Country and month

gr2 = df2.groupby(['location','month'])

# Country-wise and Month-wise total cases are added

Total_Cases = gr2['total_cases'].sum().unstack()

# Plotting heatmap to show country-wise and month-wise total cases
sns.heatmap(Total_Cases,cmap = plt.cm.Blues)
# A Figure Object is Created

fig = plt.figure(figsize = (10,5))

# A shallow copy of DataFrame is created

df3 = df2.copy()

# # New column is created by extracting only day from date

df3['day']= df3['date'].dt.day

l = ['First_Ten_Days','Mid_10_Days','Last_10_Days']

#cut days to First 10 days, Mid 10 days and Last 10 days

df3['days'] = pd.cut(df3['day'], bins = [0,10,20,31],  labels = l)

# Displaying bin win relationsip using bar plot

sns.barplot(x = 'location', y = 'month' , hue = 'days' , estimator = np.sum  ,data = df3)

plt.legend(loc = 'upper right')