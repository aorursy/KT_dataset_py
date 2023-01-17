# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv("../input/home-data-for-ml-course/train.csv")

# Call line below with no argument to check that you've loaded the data correctly
step_1.check()
# Lines below will give you a hint or solution code
#step_1.hint()
#step_1.solution()
home_data.head()
home_data.columns
# Print summary statistics in next line
home_data.describe()
unique = home_data[['Id', 'YearBuilt', 'YrSold']]
unique.describe()
last_sold_yr = home_data['YrSold'].max()                     
last_sold_mo = home_data['MoSold'].max()
latest_built = home_data['YearBuilt'].max()
latest_remodal = home_data['YearRemodAdd'].max()
print(f"Last sold on {last_sold_mo}/{last_sold_yr}")
print(f"Latest build year is {latest_built}")
print(f"Latest remodal year is {latest_remodal}")
home_data['YearBuilt'].describe()
home_data.YrSold.describe()
home_data['YearBuilt'].value_counts().sort_index(ascending=False).head()
home_data.groupby('YearBuilt').YearBuilt.value_counts().sort_values(ascending=False)
home_data.YrSold.value_counts()
home_data.groupby('YrSold').YrSold.value_counts()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.figure(figsize=(10,9))
plt.hist(home_data['YearBuilt'],bins=56)
plt.show()
plt.figure(figsize=(17,12))
sns.countplot(home_data['YearBuilt'])
sns.countplot(home_data['YrSold'])
# first graph 
minYear = min( min(home_data['YearBuilt']), min(home_data['YrSold']))
maxYear = max( max(home_data['YearBuilt']), max(home_data['YrSold']))
years = range(minYear, maxYear+1)

df = pd.DataFrame({'year': years}, index=years)
df.index.name = 'year'

df['YearBuilt'] = home_data.groupby('YearBuilt').agg({'YearBuilt': 'count'})
df['YrSold'] = home_data.groupby('YrSold').agg({'YrSold': 'count'})
df = df.drop('year', axis=1)
df = df.fillna(0)

ax1 = df.plot(kind='bar', y=['YearBuilt', 'YrSold'], figsize=(20,4), width=0.9)
def format_x(id, pos=None):
    if(years[id] % 10 == 0):
        return str(years[id])
    else:
        return ''
plt.xticks(rotation=45)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
ax1.set_title('Number of houses built and sold across the years')
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
for ax in [ax1, ax2]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
plt.show()
# second graph

import seaborn as sns; sns.set()

df = home_data.groupby(['YearBuilt', 'YrSold']).agg({'YearBuilt': 'count'})
df = df.unstack(level=1).fillna(0).T
df = df.droplevel(0)

fig, ax = plt.subplots(figsize=(20,4))  
sns.heatmap(df, cmap='Blues', ax=ax)
plt.ylabel('YearSold')
plt.yticks(rotation=0)
plt.show()
# third graph
df = home_data[['YrSold', 'SalePrice']]
fig, ax = plt.subplots(figsize=(20,4))  
sns.boxplot(y="YrSold", x="SalePrice", data=df, orient='h', ax=ax)
plt.show()
Avg_price_by_year_month = pd.DataFrame(home_data.groupby(["YrSold","MoSold"]).SalePrice.mean().round())
Cnt_price_by_year_month = pd.DataFrame(home_data.groupby(["YrSold","MoSold"]).SalePrice.count().round())

price_table = Avg_price_by_year_month.merge(Cnt_price_by_year_month,on = ["YrSold","MoSold"])
price_table = price_table.rename(index = str, columns = ({"SalePrice_x": "Avg_SalePrice", "SalePrice_y": "SaleCount" }))
price_table
price_table.plot.bar(x="Avg_SalePrice",y="SaleCount",legend=None, figsize=(20, 10), color='navy')
plt.xticks(rotation=45)
plt.title("SaleCount by Year and Month", fontsize = 20)
plt.show()
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = round(home_data['LotArea'].mean())

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = (2020-home_data['YearBuilt'].max())

# Checks your answers
step_2.check()
#step_2.hint()
#step_2.solution()