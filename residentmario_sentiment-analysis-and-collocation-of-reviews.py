import numpy as np

import pandas as pd

import seaborn as sns

pd.set_option("max_columns", None)



reviews = pd.read_csv("../input/reviews.csv")

listings = pd.read_csv("../input/listings.csv")
listings.head()
reviews.head()
listings['review_scores_rating'].sort_values().reset_index(drop=True).dropna().plot()
from IPython.display import Image



Image("https://imgs.xkcd.com/comics/star_ratings.png")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

for sentence in reviews['comments'].values[:5]:

    print(sentence)

    ss = sid.polarity_scores(sentence)

    for k in sorted(ss):

        print('{0}: {1}, '.format(k, ss[k]), end='')

    print()
# Snippet from:

# http://h6o6.com/2012/12/detecting-language-with-python-and-the-natural-language-toolkit-nltk/



from nltk.corpus import stopwords   # stopwords to detect language

from nltk import wordpunct_tokenize # function to split up our words



def get_language_likelihood(input_text):

    """Return a dictionary of languages and their likelihood of being the 

    natural language of the input text

    """

 

    input_text = input_text.lower()

    input_words = wordpunct_tokenize(input_text)

 

    language_likelihood = {}

    total_matches = 0

    for language in stopwords._fileids:

        language_likelihood[language] = len(set(input_words) &

                set(stopwords.words(language)))

 

    return language_likelihood

 

def get_language(input_text):

    """Return the most likely language of the given text

    """ 

    likelihoods = get_language_likelihood(input_text)

    return sorted(likelihoods, key=likelihoods.get, reverse=True)[0]
reviews_f = [r for r in reviews['comments'] if pd.notnull(r) and get_language(r) == 'english']
pscores = [sid.polarity_scores(comment) for comment in reviews_f]
pd.Series([score['neu'] for score in pscores]).plot(kind='hist')
pd.Series([score['pos'] for score in pscores]).plot(kind='hist')
pd.Series([score['neg'] for score in pscores]).plot(kind='hist', bins=100)
scored_reviews = pd.DataFrame()

scored_reviews['review'] = [r for r in reviews_f if get_language(r) == 'english']

scored_reviews['compound'] = [score['compound'] for score in pscores]

scored_reviews['negativity'] = [score['neg'] for score in pscores]

scored_reviews['neutrality'] = [score['neu'] for score in pscores]

scored_reviews['positivity'] = [score['pos'] for score in pscores]
scored_reviews.head()
scored_reviews.query('negativity > 0')
scored_reviews.iloc[23]['review']
scored_reviews.iloc[28]['review']
scored_reviews.query('negativity > positivity').query('negativity > 0.1')
scored_reviews.query('negativity > positivity').query('compound < -0.2')
scored_reviews.iloc[1181]['review']
scored_reviews.iloc[63836]['review']
scored_reviews.iloc[62984]['review']
scored_reviews.iloc[198]['review']
reviews_df = reviews[reviews.apply(lambda srs: pd.notnull(srs['comments']) and (get_language(srs['comments']) == 'english'), axis='columns')]
example_listing_reviews = reviews_df.query('listing_id == 1178162')
len(example_listing_reviews)
from nltk import word_tokenize
words = np.concatenate(np.array([word_tokenize(r) for r in example_listing_reviews['comments'].values]))
words
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, BigramCollocationFinder



bigram_measures = BigramAssocMeasures()

finder = BigramCollocationFinder.from_words(words)



finder.apply_freq_filter(3) 

finder.nbest(bigram_measures.pmi, 10)  
reviews_df.groupby('listing_id')['comments'].count().plot(kind='hist', bins=20)
review_words = reviews_df.groupby('listing_id').apply(

    lambda df: np.concatenate(np.array([word_tokenize(r) for r in df['comments'].values]))

)
import string



ex = ['Hi', 'there', '.', '?', '!', ',']

[w for w in ex if w not in string.punctuation]
review_words_f = review_words.map(lambda arr: np.array([w for w in arr if w not in string.punctuation]))
review_words_f.head()
def reattach_contractions(wordlist):

    words = []

    for i, word in enumerate(wordlist):

        if word[0] == "'" or word == "n't":

            words[-1] = words[-1] + word

        else:

            words.append(word)

    return words
review_words_f = review_words_f.map(reattach_contractions)
review_words_f.map(len).plot(kind='hist', bins=20)
# from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, BigramCollocationFinder



def bigramify(words):

    finder = BigramCollocationFinder.from_words(words)

    finder.apply_freq_filter(3) 

    return finder.nbest(bigram_measures.pmi, 3)



review_bigrams = review_words_f.map(bigramify)
review_bigrams.head(20)
def sample_reviews(listing_id):

    bigrams = review_bigrams[listing_id]

    review_texts = reviews[reviews['listing_id'] == listing_id]['comments'].values

    sample_reviews = []

    for bigram in bigrams:

        sample_review_list = list(filter(lambda txt: " ".join(bigram) in txt, review_texts))

        num_reviews = len(sample_review_list)

        sample_review = sample_review_list[0]

        sample_review = sample_review.replace(" ".join(bigram), "****" + " ".join(bigram) + "****")

        start_index = sample_review.index("****")

        sample_text = "..." + sample_review[start_index - 47: start_index + 47] + "..."

        sample_reviews.append(sample_text)

    return sample_reviews
listings.query('id == 3353')['listing_url']
for review in sample_reviews(3353):

    print(review)
listings.query('id == 1497879')['listing_url']
for review in sample_reviews(1497879):

    print(review)
listings.query('id == 414419')['listing_url']
for review in sample_reviews(414419):

    print(review)
listings.query('id == 1136972')['listing_url']
for review in sample_reviews(1136972):

    print(review)
for review in sample_reviews(3353):

    print(review)