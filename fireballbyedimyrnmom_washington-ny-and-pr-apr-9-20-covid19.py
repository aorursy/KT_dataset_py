#libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt #plotting, math, stats
%matplotlib inline
import seaborn as sns #plotting, regressions
USA = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
USA= USA.drop(['fips'], axis = 1) 
plt.figure(figsize=(20,7)) # Figure size
plt.title('COVID-19 cases across US states') # Title
USA.groupby("state")['cases'].max().plot(kind='bar', color='blue')
plt.figure(figsize=(20,7)) # Figure size
plt.title('COVID-19 deaths across US states') # Title
USA.groupby("state")['deaths'].max().plot(kind='bar', color='red')
NY=USA.loc[USA['state']== 'New York']
MI=USA.loc[USA['state']== 'Michigan']
WA=USA.loc[USA['state']== 'Washington']
PA=USA.loc[USA['state']== 'Pennsylvania']
PUR=USA.loc[USA['state']== 'Puerto Rico']
# Concatenate dataframes 
States=pd.concat([NY,MI,WA,PA,PUR]) 
States=States.sort_values(by=['date'], ascending=True)
plt.figure(figsize=(15,9))
plt.title('COVID-19 cases comparison of WA, IL, NY, LA, and PR') # Title
sns.lineplot(x="date", y="cases", hue="state",data=States)
plt.xticks(States.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()
plt.figure(figsize=(15,9))
plt.title('COVID-19 deaths comparison of WA, IL, NY, LA, and PR') # Title
sns.lineplot(x="date", y="deaths", hue="state",data=States, palette="cubehelix")
plt.xticks(States.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()
plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=WA, marker='o', color='blue') 
plt.title('Cases per day in Washington state') # Title
plt.xticks(WA.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()
plt.figure(figsize=(18,11))

sns.lineplot(x="date", y="cases", hue="county",data=WA, palette="Set3")
plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.title('Cases per county in Washington state') # Title
plt.show()
plt.figure(figsize=(16,7)) # Figure size
sns.lineplot(x='date', y='cases', data=NY, marker='o', color='green') 
plt.title('Cases per day in New York') # Title
plt.xticks(NY.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()
plt.figure(figsize=(22,14))
sns.lineplot(x="date", y="cases", hue="county",data=NY)
plt.xticks(NY.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.title('Cases per county in New York state') # Title
plt.show()
plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=PUR, marker='o', color='purple') 
plt.title('Cases per day in Puerto Rico') # Title
plt.xticks(PUR.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()
plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='deaths', data=PUR, marker='o', color='green') 
plt.title('Deaths per day in Puerto Rico') # Title
plt.xticks(PUR.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()
PUR