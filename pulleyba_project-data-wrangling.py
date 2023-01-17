# Downloading and importing all the necessary libraries to complete the project.

import tweepy

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import requests

import json

import os

import re

import warnings

warnings.simplefilter('ignore')
from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
pd.set_option('display.max_colwidth', -1)
archive = pd.read_csv('../input/twitter-archive-enhanced-2.csv')
from tweepy import OAuthHandler

from timeit import default_timer as timer
consumer_key = 'HIDDEN'

consumer_secret = 'HIDDEN'

access_token = 'HIDDEN'

access_secret = 'HIDDEN'



auth = OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_secret)



api = tweepy.API(auth, wait_on_rate_limit=True)



tweet_ids = archive.tweet_id.values

len(tweet_ids)
# set a function for tweet extraction

# file already created so no need to execute to continue the notebook

def tweet_extraction():

    count = 0

    fails_dict = {}

    start = timer()

    with open('tweet_json.txt', 'w') as outfile:

        for tweet_id in tweet_ids:

            count += 1

            print(str(count) + ": " + str(tweet_id))

            try:

                tweet = api.get_status(tweet_id, tweet_mode='extended')

                print("Success")

                json.dump(tweet._json, outfile)

                outfile.write('\n')

            except tweepy.TweepError as e:

                print("Fail")

                fails_dict[tweet_id] = e

                pass

    end = timer()

    print(end - start)

    print(fails_dict)
df_list = []

with open('../input/tweet_json.txt') as file:

    for line in file:

        data = json.loads(line)

        keys = data.keys()

        user = data.get('user')

        id_str = data.get('id_str')

        retweet_count = data.get('retweet_count')

        favorite_count = data.get('favorite_count')

        df_list.append({'id_str': id_str,

                        'retweet_count': retweet_count,

                        'favorite_count': favorite_count})
tweet_count = pd.DataFrame(df_list, columns = ['id_str', 'retweet_count', 'favorite_count'])
# Downloading the image predictions from the internet

folder_name = 'image_pred'

if not os.path.exists(folder_name):

    os.makedirs(folder_name)



url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'

response = requests.get(url)
with open(os.path.join(folder_name, url.split('/')[-1]), mode='wb') as file:

    file.write(response.content)
image_pred = pd.read_csv('../input/image-predictions.tsv', sep='\t')
archive.head()
archive.text.sample(20)
tweet_count.head()
image_pred.head()
archive.info()
doggo = archive.doggo.value_counts()

floofer = archive.floofer.value_counts()

pupper = archive.pupper.value_counts()

puppo = archive.puppo.value_counts()

print(doggo); 

print(floofer); 

print(pupper); 

print(puppo)
archive.name.value_counts().head(20)
archive.source.value_counts()
tweet_count.info()
image_pred.info()
# trimmed the names in order to make less wordy when coding

archive_clean = archive.copy()

image_clean = image_pred.copy()

tweet_clean = tweet_count.copy()
drop_retweet = archive_clean[pd.notnull(archive_clean['retweeted_status_id'])].index

drop_reply = archive_clean[pd.notnull(archive_clean['in_reply_to_status_id'])].index
archive_clean.drop(index=drop_retweet, inplace=True)

archive_clean.drop(index=drop_reply, inplace=True)
archive_clean.info()
archive_clean.dropna(axis='columns',how='any', inplace=True)
archive_clean.drop(columns='source', inplace=True)
archive_clean.head()
tweet_clean.rename(index=str, columns={"id_str": "tweet_id"}, inplace=True)

archive_clean.rename(columns={"floofer": "floof", 

                                         "rating_numerator": "rate_num",

                                         "rating_denominator": "rate_denom"}, inplace=True)
tweet_clean.info()
archive_clean.info()
image_clean['tweet_id'] = image_clean['tweet_id'].astype('str')

archive_clean['timestamp'] = pd.to_datetime(archive_clean['timestamp'])

archive_clean['tweet_id'] = archive_clean['tweet_id'].astype('str')
image_clean.info()
archive_clean.info()
image_clean['p1'] = image_clean['p1'].str.lower()

image_clean['p2'] = image_clean['p2'].str.lower()

image_clean['p3'] = image_clean['p3'].str.lower()
image_clean.p1.head()
image_clean.p2.head()
image_clean.p3.head()
archive_clean['text'] = archive_clean.text.str.replace("&amp;", "&")

archive_clean['text'] = archive_clean.text.str.replace("\n", " ")

archive_clean['text'] = archive_clean.text.str.replace(r"http\S+", "")

archive_clean['text'] = archive_clean.text.str.strip()
archive_clean.query("text == '&amp;'")
archive_clean.iloc[[588, 797, 853, 948, 985, 1005, 1136, 1234, 1239, 1278, 

                    1294, 1307, 1426, 1556, 1592, 1649, 1653, 1719, 1759, 

                    1811, 1860, 1922, 1960, 2005, 2014, 2047, 2076], [2,3,4,5]]
archive_clean.reset_index(inplace=True, drop=True)
archive_clean[archive_clean.text.str.contains(r"(\d+\.\d*\/\d+)")][['text', 'rate_num']]
hyphen_table = archive_clean.text.str.extractall(r"(\d+\d*\/\d+)")

hyphen_table.head(10)
match_1 = hyphen_table.query("match == 1")

match_1.head()
match_1.index.labels
# copied indices from above

archive_clean.iloc[[588, 797, 853, 948, 985, 1005, 1136, 1234, 1239, 1278, 

                    1294, 1307, 1426, 1556, 1592, 1649, 1653, 1719, 1759, 

                    1811, 1860, 1922, 1960, 2005, 2014, 2047, 2076], [2,3,4,5]]
#rating confused with 9/11(September 11th)

archive_clean.iloc[853, 3] = 14

archive_clean.iloc[853, 4] = 10



#rating confused with 4/20(Weed Day)

archive_clean.iloc[948, 3] = 13

archive_clean.iloc[948, 4] = 10



#rating confused with phrase 50/50 split

archive_clean.iloc[985, 3] = 11

archive_clean.iloc[985, 4] = 10



#rating confused with 7/11 which is name of convience store

archive_clean.iloc[1426, 3] = 10

archive_clean.iloc[1426, 4] = 10



#rating confused with 1/2 representing "half"

archive_clean.iloc[2076, 3] = 9

archive_clean.iloc[2076, 4] = 10
doubles_list = archive_clean.iloc[[588, 797, 1005, 1136, 1234, 1239, 1278, 

                    1294, 1307, 1556, 1592, 1649, 1653, 1719, 1759, 

                    1811, 1860, 1922, 1960, 2005, 2014, 2047]]

double_index = doubles_list.index
archive_clean.iloc[[41, 528, 586, 1474], [2,3,4]]
archive_clean.iloc[41, 3] = 13.5

archive_clean.iloc[528, 3] = 9.75

archive_clean.iloc[586, 3] = 11.27

archive_clean.iloc[1474, 3] = 11.26
archive_clean.iloc[[45, 528, 586, 1474], [2,3,4]]
archive_clean.iloc[[853, 948, 985, 1426, 2076], [2,3,4,5]]
doubles_list = archive_clean.iloc[[588, 797, 1005, 1136, 1234, 1239, 1278, 

                    1294, 1307, 1556, 1592, 1649, 1653, 1719, 1759, 

                    1811, 1860, 1922, 1960, 2005, 2014, 2047]]

double_index = doubles_list.index
archive_clean.drop(axis='index', index=double_index, inplace=True)
archive_clean.info()
df_merge1 = archive_clean.join(tweet_clean.set_index('tweet_id'), on='tweet_id')
df_merge1.info()
master = df_merge1.join(image_clean.set_index('tweet_id'), on='tweet_id')
master.info()
master_copy = master.copy()
drop_index = master_copy[pd.isnull(master_copy['jpg_url'])].index

drop_index2 = master_copy[pd.isnull(master_copy['retweet_count'])].index

drop_index, drop_index2
master_copy.drop(index=drop_index, inplace=True)

master_copy.drop(index=drop_index2, inplace=True)
master_copy.info()
master_copy.to_csv(path_or_buf='master.csv', index=False)
master_clean = master_copy[['timestamp','tweet_id', 'rate_num', 'rate_denom', 

                            'name', 'retweet_count', 'favorite_count', 'p1', 'p2', 'p3']]
#setting the DataFrame to index the timestamp column

master_copy = master_copy.set_index('timestamp')
master_copy.sample(5)
# viewing some descriptive statistics with the quantitative measures in our analysis. 

# used round(5) in order to visually see the last two columns without scientific notation involved.

master_copy.describe().round(5)
# this method helps to see which variables correlate and helpful when doing a hypothesis test or A/B test. 

master_copy.corr()
sns.pairplot(master_copy, vars=["rate_num", "rate_denom", "retweet_count", "favorite_count", "p1_conf", "p2_conf", "p3_conf"]);
# set these variables so the plotting code would be cleaner.

retweet_resamp = master_copy['retweet_count'].resample('1w').mean()

favorite_resamp = master_copy['favorite_count'].resample('1w').mean()
# plotting the resample of weekling favorite and retweet counts to show a smoother display over time.

sns.set(rc={'figure.figsize':(13, 6)})

fig, ax = plt.subplots()

ax.plot(retweet_resamp, marker='*', linestyle='-', linewidth=0.5, label='Retweet Count Resample')

ax.plot(favorite_resamp, marker='*', markersize=3, linestyle='--', label='Favorite Count Resample')

ax.set_ylabel('Counts')

ax.set_xlabel('Time Periods (Monthly Intervals)')

ax.legend();
tweets = np.array(master_copy.text)

my_list = []

for tweet in tweets:

    my_list.append(tweet.replace("\n",""))
mask = np.array(Image.open(requests.get('https://clipartix.com/wp-content/uploads/2016/06/Dog-bone-pink-print-dog-paw-print-transparent-background-paw-print-pink-clipart.jpg', stream=True).raw))

text = my_list
def gen_wc(text, mask):

    word_cloud = WordCloud(width = 500, height = 500, background_color='white', mask=mask).generate(str(text))

    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='red')

    plt.imshow(word_cloud)

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()
gen_wc(text, mask)