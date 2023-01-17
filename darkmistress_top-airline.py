import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import matplotlib

plt.style.use('ggplot')

from matplotlib.pyplot import figure



%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (12,8)



pd.options.mode.chained_assignment = None
data1 = pd.read_csv('../input/data101/airlines_final.csv')

data1.head(100)
# shape and data types of the data

print(data1.shape)

print(data1.dtypes)
# select numeric columns

data1_numeric = data1.select_dtypes(include=[np.number])

numeric_cols = data1_numeric.columns.values

print(numeric_cols)
# select non numeric columns

data1_non_num = data1.select_dtypes(exclude=[np.number])

non_num_cols = data1_non_num.columns.values

print(non_num_cols)
cols = data1.columns[:30] # first 30 columns

colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.

sns.heatmap(data1[cols].isnull(), cmap=sns.color_palette(colours))
# if it's a larger dataset and the visualization takes too long can do this.

# % of missing.

for col in data1.columns:

    pct_missing = np.mean(data1[col].isnull())

    print('{} - {}%'.format(col, round(pct_missing*100)))
#To detect outliers, use histogram on numeric feature

data1['wait_min'].hist(bins=100)
data1['wait_min'].describe()
#To find the rows with same value, 

#we can create a list of features with a high percentage of the same value.



num_rows = len(data1.index)

low_information_cols = [] #



for col in data1.columns:

    cnts = data1[col].value_counts(dropna=False)

    top_pct = (cnts/num_rows).iloc[0]

    

    if top_pct > 0.50:

        low_information_cols.append(col)

        print('{0}: {1:.5f}%'.format(col, top_pct*100))

        print(cnts)

        print()
#To check num of airlines



dfg = data1['airline'].value_counts(dropna=False)

dfg.describe()
#First find out which weekday has most flights

Day = data1['day'].value_counts()

Day
#Tuesday got the most flights, now we check which flight has best cleanliness...

clean = data1[(data1['cleanliness'] == 'Clean' )  & (data1['satisfaction'] == 'Very satisfied') & (data1['safety'] == 'Very safe')]

clean
dfg = clean['airline'].value_counts(dropna=False)

dfg.describe()
#Out of 33 arilines (2477 flights), we have 30 airlines (371 flights) which is 1. clean, 2. very safe and 3. very satisfactory for users

#Now lets sort our data with respect to dest_size
clean_more = clean[(clean['dest_size'] == 'Hub' )]

clean_more
clean_more.dest_region.value_counts()
import seaborn as sns
#Frist 'west US'

final_clean_west = clean_more[(clean_more['dest_region'] == 'West US')]

final_clean_west
sns.barplot(x="destination",

               y="wait_min",

               hue="airline",

               data=final_clean_west) 
final_clean_east = clean_more[(clean_more['dest_region'] == 'East US')]

final_clean_east
sns.pointplot(x='wait_min',

             y='destination',

             hue='airline',

             data=final_clean_east)
final_clean_3 = clean_more[(clean_more['dest_region'] == 'Asia')]

final_clean_3
final_clean_4 = clean_more[(clean_more['dest_region'] == 'Midwest US')]

final_clean_4
final_clean_5 = clean_more[(clean_more['dest_region'] == 'Europe')]

final_clean_5
final_clean_6 = clean_more[(clean_more['dest_region'] == 'Middle East')]

final_clean_6
final_clean_7 = clean_more[(clean_more['dest_region'] == 'Canada/Mexico')]

final_clean_7
final_clean_8 = clean_more[(clean_more['dest_region'] == 'Australia/New Zealand')]

final_clean_8
# Create figure:

fig = plt.figure(figsize=(30,19))

# Display the result - Plot 1:

plt.subplot(321)

# Plot temperatures:

sns.barplot(x='wait_min',

             y='destination',

             hue='airline',

             data=final_clean_3)

            

plt.subplot(322)

# Plot flights:

sns.scatterplot(x='wait_min',

             y='destination',

             hue='airline',

             data=final_clean_4)

            

plt.subplot(323)

sns.violinplot(x='wait_min',

             y='destination',

             hue='airline',

             data=final_clean_5)

            

plt.subplot(324)

sns.barplot(x='wait_min',

         y='destination',

         hue='airline',

         data=final_clean_6)

            

            

plt.subplot(325)

sns.barplot(x='wait_min',

            y='destination',

            hue='airline',

            data=final_clean_7)

        

plt.subplot(326)

sns.barplot(x='wait_min',

         y='destination',

         hue='airline',

         data=final_clean_8)



plt.show()
#That'all folks