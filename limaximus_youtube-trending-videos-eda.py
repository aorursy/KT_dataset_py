# import libraries



import json

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

import re



# read in all the data

ca = pd.read_csv('../input/youtube-new/CAvideos.csv', dtype={'category_id': str})

us = pd.read_csv('../input/youtube-new/USvideos.csv', dtype={'category_id': str})

uk = pd.read_csv('../input/youtube-new/GBvideos.csv', dtype={'category_id': str})
# let's load up some category data

def load_cats(json_file):

    with open(json_file) as f:

        data = json.load(f)

        

    df = json_normalize(data, 'items')

    return df



ca_cats = load_cats('../input/youtube-new/CA_category_id.json')

us_cats = load_cats('../input/youtube-new/US_category_id.json')

uk_cats = load_cats('../input/youtube-new/GB_category_id.json')



ca_cats = ca_cats.rename(columns={'snippet.title': 'category'})

us_cats = us_cats.rename(columns={'snippet.title': 'category'})

uk_cats = uk_cats.rename(columns={'snippet.title': 'category'})



print(ca_cats)
# Add a country column

ca['country'] = 'CA'

uk['country'] = 'UK'

us['country'] = 'US'
print(ca.shape)

print(uk.shape)

print(us.shape)
# merge in category ID data

ca_joined = ca.merge(ca_cats, left_on='category_id', right_on='id')

us_joined = us.merge(us_cats, left_on='category_id', right_on='id')

uk_joined = uk.merge(uk_cats, left_on='category_id', right_on='id')



ca_joined.tags.head()
#clean up the tags and descriptions

ca_joined["tags"] = ca_joined["tags"].str.replace("|", ",")

ca_joined["tags"] = ca_joined["tags"].str.replace('"', "").str.lower()

ca_joined["tags"].head()



#remove emoji

ca_joined["description"] = ca_joined["description"].str.replace("([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])", "", regex=True)

ca_joined["description"] = ca_joined["description"].str.replace(r"\\n", " ").str.lower()

ca_joined["description"].head()
#clean up the tags

us_joined["tags"] = us_joined["tags"].str.replace("|", ",")

us_joined["tags"] = us_joined["tags"].str.replace('"', "").str.lower()

us_joined["tags"].head()



#clean up descs

#remove emoji

us_joined["description"] = us_joined["description"].str.replace("([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])", "", regex=True)

us_joined["description"] = us_joined["description"].str.replace(r"\\n", " ").str.lower()

us_joined["description"].head()
#clean up the tags

uk_joined["tags"] = uk_joined["tags"].str.replace("|", ",")

uk_joined["tags"] = uk_joined["tags"].str.replace('"', "").str.lower()

uk_joined["tags"].head()



#clean up descs

#remove emoji

uk_joined["description"] = uk_joined["description"].str.replace("([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])", "", regex=True)

uk_joined["description"] = uk_joined["description"].str.replace(r"\\n", " ").str.lower()

uk_joined["description"].head()
def dedupe_sets(df):

    df_tag_sets = []

    for r in df.itertuples():

        rep_tags = r.tags.replace(' ', '_')

        things = list(set(r.tags.split(',')))

        thing_str = ','.join(things)

        df_tag_sets.append(thing_str)

    tag_sets = pd.Series(df_tag_sets)

    tag_sets.name = 'tag_sets'

    

    new_df = df.join(tag_sets)

    return new_df



uk2 = dedupe_sets(uk_joined)

ca2 = dedupe_sets(ca_joined)

us2 = dedupe_sets(us_joined)

#uk_tag_sets = []



#for r in uk_joined.itertuples():

#    things = list(set(r.tags.split(',')))

#    thing_str = ','.join(things)

#    uk_tag_sets.append(thing_str)    



#tag_sets = pd.Series(uk_tag_sets)

#tag_sets.name = 'tag_sets'

#print(tag_sets.name)



#uk2 = uk_joined.join(tag_sets)

print(uk2.head())
#export

ca_joined.to_csv('ca.csv', index=False)

us_joined.to_csv('us.csv', index=False)

uk_joined.to_csv('uk.csv', index=False)

#uk2.to_csv('uk2.csv', index=False)

#us2.to_csv('us2.csv', index=False)

#ca2.to_csv('ca2.csv', index=False)
# name the datasets and pop them in a dictionary

datasets = {'ca': ca_joined, 'uk': uk_joined, 'us': us_joined}
# put them all together and make sure the total # of rows makes sense

all_vids = pd.DataFrame()

all_vids = all_vids.append(us_joined).append(uk_joined).append(ca_joined)

all_vids.to_csv('all_vids.csv', index=False)

print(list(all_vids))

print(all_vids.shape)
print(len(all_vids.video_id.unique()))
print(len(all_vids.channel_title.unique()))
all_vids['trending_date'] = pd.to_datetime(all_vids['trending_date'], format='%y.%d.%m')

print(all_vids.trending_date.describe())

print(all_vids.trending_date.unique()) # April 8-13, 2018 are missing
# tally counts of records by category

all_vids.groupby(['country']).category.value_counts()
all_vids.head()
# Graph for Canada -- by record count

ca_counts = ca_joined.groupby('category').video_id.count().sort_values()

ca_counts.plot.barh()

plt.show()
# Graph for the US -- by record count

us_counts = us_joined.groupby('category').video_id.count().sort_values()

us_counts.plot.barh()

plt.show()
# Graph for the UK -- by record count



uk_counts = uk_joined.groupby('category').video_id.count().sort_values()

uk_counts.plot.barh()

plt.show()
us_joined.groupby(['video_id', 'title']).video_id.count().sort_values(ascending=False)