import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
allTweets = pd.read_csv('/kaggle/input/fact-checker-tweets/AllTweets_Dec2019_June2020.csv')

print(allTweets.shape)

print(allTweets.columns)

allTweets.head()
username = Counter(allTweets['username'])

print(len(username))

username.most_common(10)
# Collapsing the list

# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists

# flat_list = [item for sublist in l for item in sublist]



hashtags = list(set(allTweets['hashtags'].apply(lambda x: x.replace("[",'').replace("]", '').replace("'", ""))))

print(len(hashtags))



temp = [text.split(',') for text in hashtags]

flat_hashtag = [item.strip() for sublist in temp for item in sublist]

flat_hashtag[:10]
hashtags_C = Counter(flat_hashtag)

hashtags_C.most_common(20)