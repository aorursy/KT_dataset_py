# Import libraries
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Read data from gun-violence-data_01-2013_03-2018.csv file
data = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
data
data.columns
# Data type and summary
data.info()
data.describe()
# How many rows?
data.shape[0]
# How many columns?
data.shape[1]
data.shape
data.head()
# Values in state column
data['state'].value_counts()
data['state'].value_counts().nlargest(10)
data['n_killed'].value_counts().unique()
# Converting date calumn format to datetime format
data['date'] = pd.to_datetime(data['date'])
# Adding new columns year and month
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
# Verifying year and month columns
data.head()
data['gun_type'].value_counts()
sc = data['state'].value_counts()
sc
#sns.barplot(data=sc)
#sns.barplot(x='state', y='aa', data = sc)

#sns.distplot(x)
st = data['state'].unique()
st
#sns.barplot(data=sc)
#sns.barplot(x='state', y='aa', data = sc)
#sns.distplot(st)
cnt = data['state'].value_counts().unique()
cnt
sns.barplot(x='state', y = 'n_killed', data = data)
sns.distplot(cnt)
sns.jointplot(x='n_injured', y = 'n_killed', data = data, size=10)
#Ploting graph for year wise incedence
sns.countplot(x='year', data=data)
#Ploting graph for state wise incedence
sns.countplot(x='state', data=data)
#Violin plot for year wise killed
sns.violinplot('year', 'n_killed',data=data)
sns.distplot(data['year'], kde=False)
plt.plot(data.year)
datagrid = ['state', 'n_killed', 'n_injured', 'year']
g = sns.PairGrid(data, vars=datagrid, hue='state')
g.map_offdiag(plt.scatter)
g.add_legend()
# creating a dataframe for pie graph
state_count = data['state'].value_counts()
state_cases = pd.DataFrame({'labels':state_count.index, 'values':state_count.values})
state_cases.iloc[:20,]
plt.pie(state_cases['values'], labels = state_cases['labels'], autopct = '%1.1f%%')
# Find data for Chicago
ch = data.loc[data.city_or_county =='Chicago']
ch.head(5)
#Box Plot for n_killed or n_injured for chicago
sns.boxplot('n_killed', 'n_injured', data=ch)
#Scatter plot month wise n_killed on each year on Chicago
g = sns.FacetGrid(ch, col = "year")
g.map(plt.scatter, "month", "n_killed", alpha = .7)
g.set(xlim=(0,12))
#bar plot month wise n_killed on each year on Chicago
g = sns.FacetGrid(ch, col = "year")
g.map(sns.barplot, "month", "n_killed", alpha = 1)
g.set(xlim=(0,12))
g.set(ylim=(0,2))
g = sns.FacetGrid(ch, col = "year")
g.map(sns.barplot, "month", "n_injured")
g.set(xlim=(0,12))
#g.set(ylim=(0,2))
# density plot: killed and injured on Chicago
sns.kdeplot(ch.n_killed[ch.year ==2013], label = 'killed', shade = True)

sns.kdeplot(ch.n_injured[ch.year ==2013], label = 'injured', shade = True)
# Group by n_killed and n_injured for state
data_st = data.groupby('state').aggregate({'n_killed':np.sum, 'n_injured':np.sum}).reset_index()
data_st
# Plot graph for no of killed vs no of injured State wise.
plt.plot(data_st.state, data_st.n_killed, 'r')
plt.plot(data_st.state.head(7), data_st.n_killed.head(7), 'r' )
plt.plot(data_st.state.head(7), data_st.n_injured.head(7), 'b' )
plt.xlabel('State Name >>')
plt.suptitle('state wise case details')
sns.jointplot("n_killed","n_injured", data_st, )
