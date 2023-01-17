import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
sns.set_style("darkgrid")

import warnings
#Suppressing all warnings
warnings.filterwarnings("ignore")

%matplotlib inline

df = pd.read_csv('../input/housing-prices-in-metropolitan-areas-of-india/Mumbai.csv')
df.head()
print("Shape of the Data is",df.shape)
df.describe()
df.replace(9, np.nan, inplace=True)
df.isna().sum()
#Cleaning Location Column
location = df.Location

#Removing 'East' and 'West' from locations
for i in range(len(location)):
    last_word = location[i].split(" ")[-1]
    if "east" in last_word.lower() or "west" in last_word.lower():
        location[i] = " ".join(location[i].split(" ")[:-1])
        
#Cleaning Variations of Top Locations
def loc(location, ind):
    top_locations = location.value_counts().index[:ind]
    for i in range(len(location)):
        for j in top_locations:
            if j.lower() in location[i].lower():
                location[i] = j
    return location

location = loc(location, 36)
df['Location'] = location
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('No. of Properties by Location (Top 25)', fontsize=20)
sns.countplot(y='Location', data=df, order=df.Location.value_counts().index[:25])
ax.set_xlabel('Locations', fontsize=15)
ax.set_ylabel('No. of Properties', fontsize=15)
plt.show()
df['Area'] = pd.cut(df['Area'], bins=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, np.inf], labels=['0-250', '250-500', '500-750', '750-1000', '1000-1250', '1250-1500', '1500-1750', '1750-2000', '2000-2250', '2250-2500', '2500-2750', '2750-3000', '3000+'])
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('Area Distribution', fontsize=20)
sns.countplot(x='Area', data=df)
ax.set_ylabel('No of Properties', fontsize=15)
ax.set_xlabel('Area Ranges', fontsize=15)
plt.xticks(rotation=45)
plt.show()
df['Price'] = pd.cut(df['Price'],
                     bins=[0, 5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000, 40000000, 45000000, 50000000, 55000000, 60000000, np.inf],
                     labels=['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500', '550-600', '650-700', '700+'])

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('Price Distribution', fontsize=20)
sns.countplot(x='Price', data=df)
ax.set_ylabel('No of Properties', fontsize=15)
ax.set_xlabel('Price Ranges (In Lakhs)', fontsize=15)
plt.xticks(rotation=45)
plt.show()
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('No. of Bedrooms', fontsize=20)
sns.countplot(x='No. of Bedrooms', data=df)
ax.set_ylabel('No of Properties', fontsize=15)
ax.set_xlabel('No. of Bedrooms', fontsize=15)
plt.xticks(rotation=45)
plt.show()
def fixlabels(li):
    return ['Yes', 'No'] if li[0] == 1 else ['No', 'Yes']
fig, axes = plt.subplots(nrows=12, ncols=3, figsize=(15, 60))
for i in range(4,len(df.columns)):        
    axes[(i-4)//3][(i-4)%3].pie(df[df.columns[i]].value_counts(), labels=fixlabels(df[df.columns[i]].value_counts().index), autopct='%1.1f%%', startangle=90)
    axes[(i-4)//3][(i-4)%3].set_title(df.columns[i])
plt.show()