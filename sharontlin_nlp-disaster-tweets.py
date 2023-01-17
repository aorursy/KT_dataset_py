import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor



sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv", index_col='id')

X_test_full = pd.read_csv("../input/nlp-getting-started/test.csv", index_col='id')

X = pd.read_csv("../input/nlp-getting-started/train.csv", index_col='id')



# Separate target from predictors

y = X.target          



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
# Sentiment analysis 

import nltk

from nltk.corpus import twitter_samples

from nltk.tag import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer



positive_tweets = twitter_samples.strings('positive_tweets.json')

negative_tweets = twitter_samples.strings('negative_tweets.json')



tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

print(pos_tag(tweet_tokens[0]))
def lemmatize_sentence(tokens):

    lemmatizer = WordNetLemmatizer()

    lemmatized_sentence = []

    for word, tag in pos_tag(tokens):

        if tag.startswith('NN'):

            pos = 'n'

        elif tag.startswith('VB'):

            pos = 'v'

        else:

            pos = 'a'

        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))

    return lemmatized_sentence



print(lemmatize_sentence(tweet_tokens[0]))
# Denoising



import re, string

from nltk.corpus import stopwords



stop_words = stopwords.words('english')



def remove_noise(tweet_tokens, stop_words = ()):



    cleaned_tokens = []



    for token, tag in pos_tag(tweet_tokens):

        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\

                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)

        token = re.sub("(@[A-Za-z0-9_]+)","", token)



        if tag.startswith("NN"):

            pos = 'n'

        elif tag.startswith('VB'):

            pos = 'v'

        else:

            pos = 'a'



        lemmatizer = WordNetLemmatizer()

        token = lemmatizer.lemmatize(token, pos)



        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:

            cleaned_tokens.append(token.lower())

    return cleaned_tokens



print(remove_noise(tweet_tokens[0], stop_words))
# Sentiment denoised



positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')



positive_cleaned_tokens_list = []

negative_cleaned_tokens_list = []



for tokens in positive_tweet_tokens:

    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))



for tokens in negative_tweet_tokens:

    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    

print(positive_tweet_tokens[500])

print(positive_cleaned_tokens_list[500])
def get_all_words(cleaned_tokens_list):

    for tokens in cleaned_tokens_list:

        for token in tokens:

            yield token



all_pos_words = get_all_words(positive_cleaned_tokens_list)
from nltk import FreqDist



freq_dist_pos = FreqDist(all_pos_words)

print(freq_dist_pos.most_common(10))
def get_tweets_for_model(cleaned_tokens_list):

    for tweet_tokens in cleaned_tokens_list:

        yield dict([token, True] for token in tweet_tokens)



positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)

negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
import random



positive_dataset = [(tweet_dict, "Positive")

                     for tweet_dict in positive_tokens_for_model]



negative_dataset = [(tweet_dict, "Negative")

                     for tweet_dict in negative_tokens_for_model]



dataset = positive_dataset + negative_dataset



random.shuffle(dataset)



train_data = dataset[:7000]

test_data = dataset[7000:]
from nltk import classify

from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train_data)



print("Accuracy is:", classify.accuracy(classifier, test_data))



print(classifier.show_most_informative_features(10))
from nltk.tokenize import word_tokenize



custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."



custom_tokens = remove_noise(word_tokenize(custom_tweet))



print(classifier.classify(dict([token, True] for token in custom_tokens)))
X_train_full["keyword"].tolist()[0:100]
k_token = [a.split("%20") for a in X_train_full["keyword"].dropna(axis=0, how='all').tolist()]



print(k_token)[0:100]
X_train_full["location"].dropna(axis=0, how='all').tolist()[0:100]
X_train_full["text"].dropna(axis=0, how='all').tolist()[0:100]
# Build the model

my_model = XGBRegressor(n_estimators=200, learning_rate=0.1) 



# Fit the model

my_model.fit(X_train_full, y_train)



# Get predictions

predictions = my_model.predict(X_test) 
# Save test predictions to file

output = pd.DataFrame({'id': X_test_full.index,

                       'target': predictions})

output.to_csv('submission.csv', index=False)