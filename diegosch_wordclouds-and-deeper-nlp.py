import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



reviews = pd.read_csv('../input/Reviews.csv')

reviews.head(3)
reviews.info()



# df.info helps to easily see data format and sample size.

# Here we can see that count of non-null values is lower for Summary. 

# On a further step I'll drop rows with null to avoid processing issues.
reviews.Summary.head(10) # Preview of Summary...
# Droping null values

reviews.dropna(inplace=True) 



# The histogram reveals this dataset is highly unbalanced towards high rating. 

reviews.Score.hist(bins=5,grid=False)

plt.show()

print(reviews.groupby('Score').count().Id)
# To correct the unbalance, I'm sampling each score by the lowest n-count from above.

# (i.e. 29743 reviews scored as '2')



score_1 = reviews[reviews['Score'] == 1].sample(n=29743)

score_2 = reviews[reviews['Score'] == 2].sample(n=29743)

score_3 = reviews[reviews['Score'] == 3].sample(n=29743)

score_4 = reviews[reviews['Score'] == 4].sample(n=29743)

score_5 = reviews[reviews['Score'] == 5].sample(n=29743)



# Here I recreate a 'balanced' dataset.

reviews_sample = pd.concat([score_1,score_2,score_3,score_4,score_5],axis=0)

reviews_sample.reset_index(drop=True,inplace=True)



# Printing count by 'Score' to check dataset is now balanced.

print(reviews_sample.groupby('Score').count().Id)
# Let's now build a wordcloud on the balaced dataset looking at the 'Summary' text

# In this notebook I focus on the 'Summary' field for simplicity but the same approach

# can be applied for the 'Text' of Reviews. 



from wordcloud import WordCloud

from wordcloud import STOPWORDS



# Wordcloud function's input needs to be a single string of text.

# Here I'm concatenating all Summaries into a single string.

reviews_str = reviews_sample.Summary.str.cat()



wordcloud = WordCloud(background_color='white').generate(reviews_str)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
# Now let's split the data into Negative (Score is 1 or 2) and Positive (4 or 5) Reviews.

negative_reviews = reviews_sample[reviews_sample['Score'].isin([1,2]) ]

positive_reviews = reviews_sample[reviews_sample['Score'].isin([4,5]) ]



# Transform to single string

negative_reviews_str = negative_reviews.Summary.str.cat()

positive_reviews_str = positive_reviews.Summary.str.cat()



# Create wordclouds

wordcloud_negative = WordCloud(background_color='white').generate(negative_reviews_str)

wordcloud_positive = WordCloud(background_color='white').generate(positive_reviews_str)



# Plot

fig = plt.figure(figsize=(10,10))



ax1 = fig.add_subplot(211)

ax1.imshow(wordcloud_negative,interpolation='bilinear')

ax1.axis("off")

ax1.set_title('Reviews with Negative Scores',fontsize=20)



ax2 = fig.add_subplot(212)

ax2.imshow(wordcloud_positive,interpolation='bilinear')

ax2.axis("off")

ax2.set_title('Reviews with Positive Scores',fontsize=20)



plt.show()
# Let's create the 'Bag of Words' for Negative and Positive Reviews.

# Below I'm setting the ngram_range at (3,3) and stopwords at None to see if 

# meaningful sentences can be extracted.

# I've arbitrarely set max_features at 2000.



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



negative_vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(3, 3), 

                       max_df=1.0, min_df=1, max_features=2000,use_idf=False)

negative_vectorized = negative_vectorizer.fit_transform(negative_reviews.Summary)



positive_vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(3, 3), 

                       max_df=1.0, min_df=1, max_features=2000,use_idf=False)

positive_vectorized = positive_vectorizer.fit_transform(positive_reviews.Summary)



print(negative_vectorized.shape)

print(positive_vectorized.shape)
# Creating 3_gram term frequecy table for Negative Reviews

negative_vocab = negative_vectorizer.get_feature_names()

negative_vectorized_df = pd.DataFrame(negative_vectorized.todense(),columns=[negative_vocab]).sum()

negative_term_f = negative_vectorized_df.sort_values(ascending=False)

negative_term_f.head(30)



# More meaningful 'snippets' appear on drivers of negative reviews such as 

# Where food is manufactured: 'made in china'

# Better competitors as benchmark: 'not as good as'

# Price sensitiviy/Poor value for money: 'not worth it',' don't waste your money',

# Ingredients: 'not gluten free', 'too much sugar'
# Creating 3_gram term frequecy table for Positive Reviews



positive_vocab = positive_vectorizer.get_feature_names()

positive_vectorized_df = pd.DataFrame(positive_vectorized.todense(),columns=[positive_vocab]).sum()

positive_vectorized_df.sort_values(ascending=False).head(30)



# Not as much insight on the positives, but a lot of positive sentiment expressed.