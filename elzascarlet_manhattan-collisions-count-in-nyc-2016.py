import pandas as pd

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/database.csv')
df.head()
# Create a temp data frame

new_df = df.loc[:,['DATE','BOROUGH']]    
# Get year & month

new_df['YEAR'] = df['DATE'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y').date().strftime('%Y'))  

new_df['MONTH'] = df['DATE'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y').date().strftime('%b'))

# Filter data set to get year 2016 data only.

new_df = new_df[new_df.YEAR == '2016']         
new_df.head()
# Create a new data frame for NYC collision count.

nyc_df = new_df.loc[:,['MONTH','YEAR']]   

nyc_df = nyc_df.groupby('MONTH', sort=False).count().reset_index()      # Get collisions count

nyc_df = nyc_df.rename(columns={'YEAR':'NYC'})                          # Rename column
nyc_df.head()
new_df_m = new_df[new_df['BOROUGH'] == 'MANHATTAN']

manhattan_df = new_df_m.loc[:,['MONTH','YEAR']]

manhattan_df = manhattan_df.groupby('MONTH',sort=False).count().reset_index()

manhattan_df = manhattan_df.rename(columns={'YEAR':'MANHATTAN'})
manhattan_df.head()
monthly_df = manhattan_df.merge(nyc_df)

monthly_df.head()
monthly_df['PERCENTAGE'] = monthly_df['MANHATTAN']/monthly_df['NYC']      # Calculate percentage.

monthly_df.head()
# Generate output as csv file

monthly_df.to_csv('monthly_collision_2016.csv', index=False)
# Generate output as plot. 

manhattan = list(monthly_df['MANHATTAN'])

nyc = list(monthly_df['NYC'])

month = list(monthly_df['MONTH'])



plt.subplots(figsize=(20,8))   

sns.set_style("darkgrid")



index = np.arange(12)

bar_width = 0.5

opacity = 0.4

rects1 = plt.bar(index + bar_width/2, manhattan, bar_width, alpha=opacity, color='b', label='Manhattan')

rects2 = plt.bar(index + bar_width/2, nyc, bar_width, alpha=opacity, color='r', label='NYC')



def addlabel(rects):      # Add top labels

    for rect in rects:

        height = rect.get_height()

        label_position = height * 1.02

        plt.text(rect.get_x() + rect.get_width()/2., label_position, '%d' % int(height), ha='center', va='bottom', fontsize=15)

addlabel(rects1)

addlabel(rects2)

plt.title('Number of Collisions in Manhattan through Year 2016', fontweight='bold', fontsize=35)     

plt.xlabel('Month', fontsize=25)

plt.ylabel('Collisions Count', fontsize=25)

plt.xticks(index + bar_width, month, fontsize=20)

plt.yticks(fontsize=20)

plt.legend(fontsize=20)

plt.tight_layout()

plt.show()