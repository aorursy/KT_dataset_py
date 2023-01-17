import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
wine_df = pd.read_csv('../input/winemag/winemag-data-130k-v2.csv', index_col = 0)

prosecco_df = wine_df[wine_df['variety'] == 'Prosecco']

prosecco_df
good_prosecco = prosecco_df[prosecco_df['points'] > 89].sort_values('price', ascending = False)[['title','price','points']]

good_prosecco['title_length'] = good_prosecco['title'].map(len) 

good_prosecco
bad_prosecco = prosecco_df[prosecco_df['points'] < 85].sort_values('price', ascending = False)[['title','price','points']]

bad_prosecco['title_length'] = bad_prosecco['title'].map(len) 

bad_prosecco
print(good_prosecco['title_length'].mean())

print(bad_prosecco['title_length'].mean())
ramen_df = pd.read_csv('../input/ramen-rating/ramen-ratings.csv')

ramen_df.loc[ramen_df.Stars == 'Unrated', 'Stars'] = np.nan

ramen_df.Stars.fillna(ramen_df.mean())

ramen_df.Stars = pd.to_numeric(ramen_df.Stars)
ramen_df.groupby('Country').agg(mean = ('Stars','mean'),

                                q10 = ('Stars',lambda x: x.quantile(0.1)),

                                q90 = ('Stars',lambda x: x.quantile(0.9)))
ramen = pd.read_csv('../input/ramen-rating/ramen-ratings.csv')

ramen
ramen.groupby(['Country']).Style.value_counts(normalize = True)
order_df = pd.read_csv('../input/restaurants/orders.csv')

customers_df = pd.read_csv('../input/restaurants/customers.csv')

#order_df.set_index('customer_id', inplace = True)

order_df
customers_df.rename(columns = {'akeed_customer_id' : 'customer_id'}, inplace = True)

customers_df

#customers_df.set_index('customer_id', inplace = True)
# I didnt manage merge the dataframes using join

restaurant_merged = pd.merge(customers_df,order_df,on ='customer_id')[['item_count','gender']]

restaurant_merged
print(restaurant_merged.gender.value_counts())

print(restaurant_merged.gender.unique())
#trim all strings and fix lower case male entries

restaurant_merged.gender = restaurant_merged.gender.str.strip()

restaurant_merged.loc[restaurant_merged.gender =='male','gender'] = 'Male'

# A few values are empty strings, I change them to nan

restaurant_merged.loc[restaurant_merged.gender.eq(''),'gender'] = np.nan

#Find the probability distribution among genders

gender_prob_dist = restaurant_merged.gender.value_counts(normalize = True)

#Create array with Male and Female values based on prob dist

fill = np.random.choice(['Male','Female'],

                     p=[gender_prob_dist[0], gender_prob_dist[1]], 

                     size = len(restaurant_merged[restaurant_merged.gender.isna()]))

#Fill all nan values with prob dist array

restaurant_merged.loc[restaurant_merged.gender.isna(),'gender'] = fill

#fill item_count nan with median value

restaurant_merged.item_count.fillna(restaurant_merged.item_count.median(), inplace = True)

restaurant_merged
restaurant_merged.groupby('gender').item_count.mean()
billboard = pd.read_csv('../input/billboard/billboard.csv')

billboard
billboard = pd.melt(billboard, id_vars = billboard.columns[:7], value_vars =billboard.columns[7:], var_name = 'week', value_name = 'Rank' )

billboard
billboard.week = billboard.week.str.extract('(\d+)')

billboard
billboard.groupby(['artist.inverted','track']).Rank.mean().sort_values().head(10)