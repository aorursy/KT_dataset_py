## importing all Libraries including matplotlib and seaborn.
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt   
import seaborn as sns

## Reading the CSV file.
GV = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
GV.columns
## Observing Data types for the Dataset
GV.info()
GV.describe() 
#GV.dtypes
GV.head()
GV.shape
## All Below Columns have multiple values.
#GV['gun_type'].value_counts()        # Distribution
#GV['gun_type'].unique()  
#GV['participant_status'].unique() 
#GV['n_guns_involved'].unique()
#GV['city_or_county'].unique()
GV['city_or_county'].value_counts().nlargest(10) # This gives top 10 values.
# Chicago has max cases.
GV['state'].value_counts().nlargest(10) # This gives top 10 values
#GV.state.value_counts(ascending=False)
## Converting date column to datetime format for splitting into year and month column.
GV['date'] = pd.to_datetime(GV['date'])  
## Adding New Column as year, extracting it from date column
GV['year'] = GV['date'].dt.year
## Adding New Column as month, extracting it from date column
GV['month'] = GV['date'].dt.month
## Checking data and columns 
GV.head()
pd.unique(GV['gun_type']) # How manyn Gun types.
#GV['gun_type1'] = GV.gun_type.split['||']  # Need to check how to split
## Plotting graph for Year wise Gun violence Incidence/Cases..
sns.countplot(x='year', data=GV) 
## Cases are Increasing from 2014 onwards.
## Cases reported monthwise from 2013 - 2018. Max Cases in Jan.
sns.countplot(x='month', data=GV)
## Violin plot for year wise n killed. 2016 has highest.
sns.violinplot("year", "n_killed", data=GV );
## Graph for state wise cases
sns.countplot(x='state', data=GV)
## Plotting graph for Year wise Gun Violence Incidence.
g = sns.distplot(GV['year'], kde=False);
plt.plot(GV.year)
datagrid = ['state', 'n_killed', 'n_injured', 'year']
g = sns.PairGrid(GV,
                 vars=datagrid,  # Variables in the grid
                 hue='state'       # Variable as per which to map plot aspects to different colors.
                 )
#histogram
#g = g.map_diag(plt.hist)                   

#g.map_offdiag(plt.scatter)
g.map_offdiag(plt.scatter)

g.add_legend();
#g.add_legend();
## Creating a DataFrame for plotting pie graph
state_count = GV['state'].value_counts()

state_cases = pd.DataFrame({'labels': state_count.index,
                                   'values': state_count.values })

state_cases.iloc[:20,] # Just showing first twenty rows.
#state_cases.head(20)
plt.pie(
    state_cases['values'],
    # creating pie diagrams with values and labels
    labels=state_cases['labels'],
    autopct='%1.1f%%',
    )
## Getting Data for Chicago..
ch = GV.loc[GV.city_or_county == 'Chicago']
ch.head(5)
## Box Plot for n_killed or n_injured for Chicago data.
sns.boxplot("n_killed", "n_injured", data= ch)
g = sns.FacetGrid(ch, col="year")
g.map(plt.scatter, "month", "n_killed", alpha=.7)
g.set(xlim=(0,12))
#g.add_legend();
g = sns.FacetGrid(ch, col="year")
g.map(sns.barplot, "month", "n_killed")
g.set(xlim=(0,12))
#g.set(ylim=(0,5))

### Need to know why n_killed is shown as 0.2, 0.4 etc???, and if xlim is not set - it shows for 1,2,3 month only
g = sns.FacetGrid(ch, col="year")
g.map(sns.barplot, "month", "n_injured")
g.set(xlim=(0,12))
#FacetGrid for n_injured.

# Density plot: sns.kdeplot() from ch (Chicago) dataset. killed and injured for 2013

sns.kdeplot(ch.n_killed[ch.year == 2013], label='killed', shade=True)

sns.kdeplot(ch.n_injured[ch.year == 2013], label='injured', shade=True)
## Group by n_killed and n_injured for State
GV_C = GV.groupby('state').aggregate({'n_killed': np.sum, 'n_injured': np.sum}).reset_index()

### Initially did without .reset_index --> state column couldnt be used.
GV_C.head()
## Plot Graph for no of killed vs no of injured (Total) State wise.
# Red Line shows as killed
#plt.figure()
plt.plot(GV_C.head(7).state,GV_C.head(7).n_killed, 'r')
plt.plot(GV_C.head(7).state,GV_C.head(7).n_injured)
plt.xlabel('State Name')
plt.suptitle('State wise Cases detail for first 7 states from GV_C')

## Joint Plot for n_killed and n_injured
sns.jointplot("n_killed","n_injured", GV_C, kind='hex')
## Thanks