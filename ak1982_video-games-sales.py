import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read the data
df = pd.read_csv('../input/videogamesales/vgsales.csv')
# check few rows
df.head()
#Count of how many row and columns
df.shape

#Check for data types
df.info()
missing_values_perct = df.isna().sum()*100/df.shape[0]
missing_values_perct
df.dropna(inplace = True)

#confirm missin values are dropped
df.isna().sum()
top100 = df.head(100)
plt.figure(figsize=(10,5))
ax = sns.swarmplot(x = 'Publisher',y = 'Global_Sales', data = top100, alpha = 0.8).set_title("Top 100 globally sold games")
plt.xticks(rotation = 90)
plt.figure(figsize=(15,8))
sns.set_style('darkgrid')
ax= sns.countplot(x = 'Genre', data = df,  order = df['Genre'].value_counts().index).set_title('Top Genre Games ')
plt.xticks(rotation =90)
group_by_genre = df.groupby('Genre').sum().loc[:,'NA_Sales':'Other_Sales']
plt.figure(figsize=(15,10))
sns.set_style('darkgrid')

ax = sns.heatmap(group_by_genre,annot=True,fmt = '.1f').set_title('Comparision for each genre and region')

plt.figure(figsize=(15,10))
sns.set_style('darkgrid')
ax = sns.barplot(x = 'Platform',y = 'Global_Sales', data = top100, ci = None, palette = 'bright').set_title("Top 100 globally sold games on platform")
plt.xticks(rotation = 90)


plt.figure(figsize=(15,10))
sns.set_style('darkgrid')
sns.lineplot(x = 'Year', y = 'NA_Sales', data = df, color = 'red',ci = None, label = 'North America')
sns.lineplot(x = 'Year', y = 'EU_Sales', data = df, color = 'blue',ci = None,label = 'Europe')

plt.ylabel('Sales')
plt.title("Sales comparison North America VS Europe")





plt.figure(figsize=(15,10))
sns.set_style('darkgrid')

yearly_sale = df[['Year','Global_Sales']].groupby('Year').sum().reset_index()
yearly_sale['Year'] = yearly_sale['Year'].astype(int)

sns.barplot(x = 'Year',y= 'Global_Sales',data = yearly_sale).set_title('Global total sale yearly')
plt.xticks(rotation =90)