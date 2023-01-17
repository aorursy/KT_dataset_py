import pandas as pd
from pandas.api.types import CategoricalDtype
import statsmodels.api as sm
import numpy as np
import pylab as plt
import json
tweets_df = pd.read_json("../input/tweets.json").set_index('date')
tweets_df.columns = ['tweets']

twitter_mentions_df = pd.read_json("../input/twitter_mentions.json").set_index('date')
twitter_mentions_df.columns = ['twitter_mentions']

mood_df = pd.read_json("../input/mood.json").set_index('date')
mood_df.columns = ['mood_int']
mapper = {1: 'Terrible', 
          2: 'Bad',
          3: 'Okay',
          4: 'Good',
          5: 'Great'}
mood_df['mood_cat'] = mood_df['mood_int'].replace(mapper)
mood_df['mood_cat'] = mood_df['mood_cat'].astype(CategoricalDtype(ordered=True))
mood_df['mood_cat'] = mood_df['mood_cat'].cat.reorder_categories(mapper.values(), ordered=True)
combined_df = mood_df.join(tweets_df, how='outer').join(twitter_mentions_df, how='outer')
plt.figure(figsize=(24,5))
plt.plot(combined_df.mood_int, label="mood")
combined_df['mood_baseline'] = 3
combined_df.loc[combined_df.index < '2017-05-10', 'mood_baseline'] = 4
plt.figure(figsize=(24,5))
plt.plot(combined_df.mood_int, label="mood_int")
plt.plot(combined_df.mood_baseline, label="mood_baseline")
plt.legend()
combined_df.describe()
combined_df.to_csv('combined_measures.csv', index=True)