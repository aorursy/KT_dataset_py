# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read in dataset
import pandas as pd
apps_with_duplicates = pd.read_csv('/kaggle/input/google-android-apps/apps.csv')
print(apps_with_duplicates.shape)

# Drop duplicates
apps = apps_with_duplicates.drop_duplicates()
print(apps.shape)
#output shows there were no duplicate columns
# Print the total number of apps
print('Total number of apps in the dataset = ', len(apps))

# Print a concise summary of apps dataframe
print(apps.info())

# Have a look at a random sample of n rows
n = 5
apps.sample(n)
# List of characters to remove
chars_to_remove = ['+',',','$']
# List of column names to clean
cols_to_clean = ['Installs','Price']

# Loop for each column
for col in cols_to_clean:
    # Replace each character with an empty string
    for char in chars_to_remove:
        #.astype() is for type conversion, here apps col is being converted to sting type
        #.str.replace(char, '') it works on a sting(.str signifies this) which contains char and replaces the char by'' 
        apps[col] = apps[col].astype(str).str.replace(char, '')
    # Convert col to numeric .to_numeric(apps[col]) converts apps[col back to numeric type]
    apps[col] = pd.to_numeric(apps[col])
apps.info()  
#from output , it can be seen that price and installs are numeric type now 
import plotly
#this next line makes sure that plots are shown in jupyter notebook when internet is disconnected...offline.
plotly.offline.init_notebook_mode(connected=True)
#graph_objs :This package imports definitions for all of Plotly's graph objects
import plotly.graph_objs as go

# Print the total number of unique categories, unique fn return total no. of unique category in apps.
print(apps['Category'].unique())
num_categories = len(apps['Category'].unique())
print('Number of categories = ', num_categories)

# Count the number of apps in each 'Category' and sort them in descending order
num_apps_in_category = apps['Category'].value_counts().sort_values(ascending = False)
print(num_apps_in_category)
# print(type(num_apps_in_category))
# output :it is pandas.core.series.series
data = [go.Bar(
        x = num_apps_in_category.index, # index = category name
        y = num_apps_in_category.values, # value = count
)]
ylabel= 'numper of apps '
# similar to plt.show of matplotlib
plotly.offline.iplot(data)

# Average rating of apps
avg_app_rating = apps['Rating'].mean()
print('Average app rating = ', avg_app_rating)

# Distribution of apps according to their ratings
data = [go.Histogram(
        x = apps['Rating']
)]

# Vertical dashed line to indicate the average app rating
layout = {'shapes':[ {
              'type' :'line',
              'x0': avg_app_rating,
              'y0': 0,
              'x1': avg_app_rating,
              'y1': 1000,
    # "dash" in next line is a keyword argumnet whose value is dashdot
              'line': { 'dash': 'dashdot'}
          }]
          }

plotly.offline.iplot({'data': data, 'layout': layout})
%matplotlib inline
import seaborn as sns
sns.set_style("darkgrid")
import warnings #used to remove warnings
warnings.filterwarnings("ignore")

# Filter rows where both Rating and Size values are not null, ~ is used for 'not'
apps_with_size_and_rating_present = apps[(~apps['Rating'].isnull()) & (~apps['Size'].isnull())]
print(apps_with_size_and_rating_present.head(6))

# Subset for categories with at least 250 apps
large_categories = apps_with_size_and_rating_present.groupby('Category').filter(lambda x: len(x) >= 250).reset_index()

# Plot size vs. rating
plt1 = sns.jointplot(x = large_categories['Size'], y = large_categories['Rating'], kind = 'hex')
#here keyword argument kind = hex makes hexagons in scatter plot

# Subset apps whose 'Type' is 'Paid'
paid_apps = apps_with_size_and_rating_present[apps_with_size_and_rating_present['Type'] == 'Paid']

# Plot price vs. rating
plt2 = sns.jointplot(x = paid_apps['Price'], y = paid_apps['Rating'])
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

# Select a few popular app categories
popular_app_cats = apps[apps.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY',
                                            'MEDICAL', 'TOOLS', 'FINANCE',
                                            'LIFESTYLE','BUSINESS'])]
print(popular_app_cats.head(10))

# Examine the price trend by plotting Price vs Category
ax = sns.stripplot(x = popular_app_cats['Price'], y = popular_app_cats['Category'], jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories')

# Apps whose Price is greater than 200
apps_above_200 = popular_app_cats[['Category', 'App', 'Price']][popular_app_cats['Price'] > 200]
apps_above_200
# Select apps priced below $100
apps_under_100 = popular_app_cats[popular_app_cats['Price']<100]

fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

# Examine price vs category with the authentic apps (apps_under_100)
ax = sns.stripplot(x= 'Price', y='Category', data=apps_under_100,
                   jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories after filtering for junk apps')
trace0 = go.Box(
    # Data for paid apps
    y=apps[apps['Type'] == 'Paid']['Installs'],
    name = 'Paid'
)

trace1 = go.Box(
    # Data for free apps
    y=apps[apps['Type'] == 'Free']['Installs'],
    name = 'Free'
)

layout = go.Layout(
    title = "Number of downloads of paid apps vs. free apps",
    yaxis = dict(
        type = 'log',
        
    )
)

# Add trace0 and trace1 to a list for plotting in a single plot
data = [trace0, trace1]
plotly.offline.iplot({'data': data, 'layout': layout})
# Load user_reviews.csv
reviews_df = pd.read_csv('/kaggle/input/google-android-app-reviews/googleplaystore_user_reviews.csv')

# Join and merge the two dataframe,how= inner ,inner join is a type of join.
merged_df = pd.merge(apps, reviews_df, on = 'App', how = "inner")


# Drop NA values from Sentiment and Translated_Review columns
merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])

#this statement is of no use :sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)

# User review sentiment polarity for paid vs. free apps
ax = sns.boxplot(x = merged_df.Type, y = merged_df.Sentiment_Polarity, data = merged_df)
ax.set_title('Sentiment Polarity Distribution')
merged_df





