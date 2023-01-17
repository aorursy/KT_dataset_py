import pandas as pd

import matplotlib.pyplot as plt

from datetime import timedelta

import numpy as np

import seaborn as sns

sns.set()
df = pd.read_csv("../input/ecommerce-bookings-data/ecommerce_data.csv")

df.head(10)
df.dtypes
df.isnull().sum()
df.hist(figsize = (10,6))

plt.subplots_adjust(hspace = 1, wspace = 1, top = 0.9)

plt.tight_layout()

plt.suptitle('Histograms of Features')
fig, ax = plt.subplots(2,2, figsize = (10,6))

fig.tight_layout()

fig.suptitle('Comparison of Distributions below Specific Quantiles of `order`')

fig.subplots_adjust(top = 0.85, hspace = 0.3, wspace = 0.3)

j = 0

i = 0

quantile_list = [0.8, 0.9, 0.95, 0.99]

for q in quantile_list:

    ax[i,j].hist(df['orders'][df['orders']<df['orders'].quantile(q)])

    ax[i,j].set_title("Below {}$^{{th}}$ Percentile".format(int(q*100)))

    if j < 1:

        j+=1

    else:

        i+=1

        j-=1
df['orders'].describe().round(1)
cutoff = df['orders'].quantile(0.95)

# Create a dummy indicating if the number of orders was greater than 95th percentile

df['high_volume'] = (df['orders']>cutoff)*1

df.loc[(df['orders']>cutoff), 'orders'] = cutoff

df['high_volume'].value_counts(normalize = True)
df['date'] = pd.to_datetime(df['date'])
pd.crosstab(df['date'], df['city_id']).head(20)
# Defines a function to identify which dates are missing for each city

def missing_date_info(city, data):

    city_df = data.loc[data['city_id']==city, :] #create a temporary dataframe for a given city



    base = city_df['date'].min() #identifies the earliest date of observation

    maxdate = city_df['date'].max() #identifies the latest date of observation

    full_date_list = [base + timedelta(days=x) for x in range((maxdate-base).days)] # creates a list of dates spanning from earliest to latest that includes every day in between



    date_vals = sorted(set(city_df['date'])) #create a list of the date values currently in the dataframe for a given city

    date_missing = [i for i in full_date_list if i not in date_vals] #identifies the dates that are not currently represented in the dataframe

    return date_missing
missing_dict = {i : [] for i in df.columns}

for x in range(df['city_id'].max()+1):

    dates = missing_date_info(x, df) #gets the missing dates for each city   

    #Appends information to the new dictionary

    for d in dates:

        missing_dict['date'].append(d)

        missing_dict['product_id'].append(np.nan)

        missing_dict['city_id'].append(x)

        missing_dict['orders'].append(0)

        missing_dict['high_volume'].append(0)



# Creates a new dataframe with only the missing date rows + other columns    

missing_rows = pd.DataFrame(missing_dict)
missing_rows
# Appends missing rows to the original dataframe and sorts the dataframe by city and date

df2 = df.append(missing_rows)

df2 = df2.sort_values(['city_id', 'date']).reset_index(drop = True)
print('df had {} rows.\nAnd df2 now has {} rows.'.format(len(df), len(df2)))
test_list = []

for x in range(df2['city_id'].max()+1):

    test_list.append(missing_date_info(x, df2) == [])

print('Are all of the missing date lists empty?\nResponse: {}'.format(all(test_list)))
df2.to_csv("ecommerce_clean.csv", index = False)