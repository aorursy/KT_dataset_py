from kaggle.competitions import twosigmanews
import numpy as np
from tqdm import tqdm

# Get data
env = twosigmanews.make_env()
market_train_df, news_train_df = env.get_training_data()
market_train_df.head()
news_train_df.head()
# # Get set of subjects
# subjectSet = set()
# for row in tqdm(news_train_df['subjects']):
#     myset = eval(row)
#     subjectSet = subjectSet.union(myset)
# # Convert subjects to columns
# for subject in tqdm(subjectSet):
#     news_train_df[subject] = news_train_df["subjects"].str.contains(subject)
news_train_df['date'] = news_train_df['time'].dt.date
market_train_df['date'] = market_train_df['time'].dt.date
%%time

# Make news_train_df smaller
news_small = news_train_df.drop(["time", "sourceTimestamp", "firstCreated",
                               "sourceId", "headline", "takeSequence",
                               "provider", "subjects", "audiences",
                               "companyCount", "marketCommentary", "assetCodes"], axis=1)
news_small.head()
%%time
# multiply columns by relevance
weighted_cols = ['urgency', 'bodySize', 'sentenceCount',
                 'wordCount', 'firstMentionSentence', 'sentimentClass',
                 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive', 
                 'sentimentWordCount', "noveltyCount12H", "noveltyCount24H",
                 "noveltyCount3D", "noveltyCount5D", "noveltyCount7D",
                 "volumeCounts12H", "volumeCounts24H", "volumeCounts3D",
                 "volumeCounts5D", "volumeCounts7D"]

# memory error if we use all columns at once
for col in weighted_cols:
    news_small[col] = news_small[col] * news_small['relevance']
%%time
# sum all columns by group, now relevance becomes total relevance
sumFunctions = {"relevance": np.sum,
                "urgency": np.sum,
                "bodySize": np.sum,
                "sentenceCount": np.sum,
                "wordCount": np.sum,
                "firstMentionSentence": np.sum,
                "sentimentClass": np.sum,
                "sentimentNegative": np.sum,
                "sentimentNeutral": np.sum,
                "sentimentPositive": np.sum,
                "sentimentWordCount": np.sum,
                "noveltyCount12H": np.sum,
                "noveltyCount24H": np.sum,
                "noveltyCount3D": np.sum,
                "noveltyCount5D": np.sum,
                "noveltyCount7D": np.sum,
                "volumeCounts12H": np.sum,
                "volumeCounts24H": np.sum,
                "volumeCounts3D": np.sum,
                "volumeCounts5D": np.sum,
                "volumeCounts7D": np.sum}
news_small = news_small.groupby(["date","assetName"]).agg(sumFunctions)

# divide everything by total relevance to get weighted averages
for col in weighted_cols:
    news_small[col] = news_small[col] / news_small['relevance']

import pandas as pd
# now we merge market and news, also drop relevance since it is already used
news_small = news_small.drop('relevance', axis=1)
df = pd.merge(market_train_df, news_small, how='left', on=['date', 'assetName'])
df.head()
