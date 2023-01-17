import pandas as pd
import numpy as np
from textblob import TextBlob
Trump_reviews = pd.read_csv('../input/us-election-using-twitter-sentiment-analysis/US Election using twitter sentiment/Trumpall2.csv', encoding = 'utf-8')
Biden_reviews = pd.read_csv('../input/us-election-using-twitter-sentiment-analysis/US Election using twitter sentiment/Bidenall2.csv', encoding = 'utf-8')
Trump_reviews.head()
Biden_reviews.head()
Trump_reviews['text'][10]
Biden_reviews['text'][500]
text_blob_object1 = TextBlob(Trump_reviews['text'][10])
print(text_blob_object1.sentiment)
text_blob_object2 = TextBlob(Biden_reviews['text'][500])
print(text_blob_object2.sentiment)
text_blob_object2 = TextBlob(Biden_reviews['text'][100])
print(text_blob_object2.sentiment)
def find_pol(review):
    return TextBlob(review).sentiment.polarity

Trump_reviews['Sentiment_Polarity'] = Trump_reviews['text'].apply(find_pol)
Trump_reviews.tail()
def find_pol(review):
    return TextBlob(review).sentiment.polarity

Biden_reviews['Sentiment_Polarity'] = Biden_reviews['text'].apply(find_pol)
Biden_reviews.tail()
reviews1 = Trump_reviews[Trump_reviews['Sentiment_Polarity'] == 0.0000]
reviews1.shape
reviews2 = Biden_reviews[Biden_reviews['Sentiment_Polarity'] == 0.0000]
reviews2.shape
cond1 = Trump_reviews['Sentiment_Polarity'].isin(reviews1['Sentiment_Polarity'])
Trump_reviews.drop(Trump_reviews[cond1].index, inplace = True)
Trump_reviews.shape 
cond2 = Biden_reviews['Sentiment_Polarity'].isin(reviews1['Sentiment_Polarity'])
Biden_reviews.drop(Biden_reviews[cond2].index, inplace = True)
Biden_reviews.shape
# Let's make both the datasets balanced now. So we will just take 1000 rows from both datasets and drop rest of them.

np.random.seed(10)
remove_n =324
drop_indices = np.random.choice(Trump_reviews.index, remove_n, replace=False)
df_subset_trump = Trump_reviews.drop(drop_indices)
df_subset_trump.shape
np.random.seed(10)
remove_n =31
drop_indices = np.random.choice(Biden_reviews.index, remove_n, replace=False)
df_subset_biden = Biden_reviews.drop(drop_indices)
df_subset_biden.shape