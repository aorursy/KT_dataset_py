import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz
# Load in the dataset
ramen = pd.read_csv('/kaggle/input/ramen-ratings-latest-update-jan-25-2020/Ramen_ratings_2020.csv')

# Display the first columns
ramen.head()
ramen.drop('URL', axis=1, inplace=True)
# Check data type and null values
ramen.info()
# Remove leading and trailing spaces in each string value
for col in ramen[['Brand','Variety','Style','Country']]:
    ramen[col] = ramen[col].str.strip()
    print('Number of unique values in ' + str(col) +' is ' + str(ramen[col].nunique()))
ramen['Country'].unique()
brand_country = ramen['Brand'] +' '+ ramen['Country']
brand_country.nunique()
unique_brand = ramen['Brand'].unique().tolist()
sorted(unique_brand)[:20]
process.extract('7 Select', unique_brand, scorer=fuzz.token_sort_ratio)
process.extract('A-Sha', unique_brand, scorer=fuzz.token_sort_ratio)
process.extract('Acecook', unique_brand, scorer=fuzz.token_sort_ratio)
process.extract("Chef Nic's Noodles", unique_brand, scorer=fuzz.token_sort_ratio)
process.extract('Chorip Dong', unique_brand, scorer=fuzz.token_sort_ratio)
process.extract('7 Select', unique_brand, scorer=fuzz.token_set_ratio)
process.extract('A-Sha', unique_brand, scorer=fuzz.token_set_ratio)
process.extract('Acecook', unique_brand, scorer=fuzz.token_set_ratio)
process.extract("Chef Nic's Noodles", unique_brand, scorer=fuzz.token_set_ratio)
process.extract('Chorip Dong', unique_brand, scorer=fuzz.token_set_ratio)
#Create tuples of brand names, matched brand names, and the score
score_sort = [(x,) + i
             for x in unique_brand 
             for i in process.extract(x, unique_brand, scorer=fuzz.token_sort_ratio)]
#Create dataframe from the tuples
similarity_sort = pd.DataFrame(score_sort, columns=['brand_sort','match_sort','score_sort'])
similarity_sort.head()
#Derive representative values
similarity_sort['sorted_brand_sort'] = np.minimum(similarity_sort['brand_sort'], similarity_sort['match_sort'])
similarity_sort.head()
high_score_sort = similarity_sort[(similarity_sort['score_sort'] >= 80) &
                                      (similarity_sort['brand_sort'] != similarity_sort['match_sort']) &
                                      (similarity_sort['sorted_brand_sort'] != similarity_sort['match_sort'])]
#Drop the representative value column
high_score_sort = high_score_sort.drop('sorted_brand_sort',axis=1).copy()
#Group matches by brand names and scores
#pd.set_option('display.max_rows', None)
high_score_sort.groupby(['brand_sort','score_sort']).agg(
                        {'match_sort': ', '.join}).sort_values(
                        ['score_sort'], ascending=False)
#Souper - Super - 91%
ramen[(ramen['Brand'] == 'Souper') | (ramen['Brand'] == 'Super')].sort_values(['Brand'])
#Sura - Suraj - 89%
ramen[(ramen['Brand'] == 'Sura') | (ramen['Brand'] == 'Suraj')].sort_values(['Brand'])
#Ped Chef - Red Chef - 88%
ramen[(ramen['Brand'] == 'Ped Chef') | (ramen['Brand'] == 'Red Chef')].sort_values(['Brand'])
#Create tuples of brand names, matched brand names, and the score
score_set = [(x,) + i
             for x in unique_brand 
             for i in process.extract(x, unique_brand, scorer=fuzz.token_set_ratio)]
#Create dataframe from the tuples and derive representative values
similarity_set = pd.DataFrame(score_set, columns=['brand_set','match_set','score_set'])
similarity_set['sorted_brand_set'] = np.minimum(similarity_set['brand_set'], similarity_set['match_set'])

#Pick values
high_score_set = similarity_set[(similarity_set['score_set'] >= 80) & 
                                    (similarity_set['brand_set'] != similarity_set['match_set']) & 
                                    (similarity_set['sorted_brand_set'] != similarity_set['match_set'])]

#Drop the representative value column
high_score_set = high_score_set.drop('sorted_brand_set',axis=1).copy()
#Group brands by matches and scores
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
high_score_set.groupby(['match_set','score_set']).agg(
                       {'brand_set': ', '.join}).sort_values(
                       ['score_set'], ascending=False)
#Create columns with brand names combining scores
high_score_sort['brand_sort'] = high_score_sort['brand_sort'] + ': ' + high_score_sort['score_sort'].astype(str)
high_score_set['brand_set'] = high_score_set['brand_set'] + ': ' + high_score_set['score_set'].astype(str)
#Group data by matched name and store in new dataframe
token_sort = high_score_sort.groupby(['match_sort']).agg({'brand_sort': ', '.join}).reset_index()
token_set = high_score_set.groupby(['match_set']).agg({'brand_set': ', '.join}).reset_index()

#Rename columns
token_sort = token_sort.rename(columns={'match_sort':'brand'})
token_set = token_set.rename(columns={'match_set':'brand'})
#Outer join two tables by brand (matched names)
similarity = pd.merge(token_sort, token_set, how='outer', on='brand')

#Replace NaN values and rename columns for readability
similarity = similarity.replace(np.nan,'')
similarity = similarity.rename(columns={'brand_set':'token_set_ratio','brand_sort':'token_sort_ratio'})
similarity.sort_values('brand')