import numpy as np

import pandas as pd



from nltk import FreqDist

from nltk.corpus import twitter_samples, stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tag import pos_tag

from nltk.tokenize import TweetTokenizer



from nltk import classify, NaiveBayesClassifier



import pickle



import re, string
positive = pd.DataFrame(twitter_samples.strings('positive_tweets.json'), columns=['tweet'])

negative = pd.DataFrame(twitter_samples.strings('negative_tweets.json'), columns=['tweet'])



positive['sentiment'] = 'pos'

negative['sentiment'] = 'neg'



df = pd.concat([positive, negative])



# With the method .sample(frac=1), the whole dataframe is sampled. It's an easy way to shuffle rows.

# The method .reset_index(drop=True) is to avoid keeping a column with the former indices.

df = df.sample(frac=1).reset_index(drop=True)



df.head()
tweet_tokenizer = TweetTokenizer()



df['tokens'] = df['tweet'].apply(lambda x : tweet_tokenizer.tokenize(x))



df.head()
stop_words = stopwords.words('english')



def remove_noise(tokens):

    

    # Given a list of tokens, a list of clean tokens is returned.

    clean_tokens = []

    

    lemmatizer = WordNetLemmatizer()

    

    for token, tag in pos_tag(tokens):

        

        # If the token has any symbol which is not alphanumerical, it is removed.

        token = re.sub('.*[^\w].*', '', token)

        

        # This logic identifies the 'part of speech' (pos) of the token.

        if tag.startswith('NN'):

            # Noun.

            pos = 'n'

        elif tag.startswith('VB'):

            # Verb.

            pos = 'v'

        elif tag.startswith('JJ'):

            # Adjective.

            pos = 'a'

        else:

            continue

        

        # The token is lemmatized.

        token = lemmatizer.lemmatize(token, pos)

        

        # The token is or is not added to the list.

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:

            clean_tokens.append(token.lower())

        

    return clean_tokens
df['clean_tokens'] = df['tokens'].apply(remove_noise)



df.head()
def get_all_words(tokens_list):

    for tokens in tokens_list:

        for token in tokens:

            yield token
mask_positive = df['sentiment'] == 'pos'

mask_negative = df['sentiment'] == 'neg'



all_positive_tokens = get_all_words(df.loc[mask_positive, 'clean_tokens'].to_list())

all_negative_tokens = get_all_words(df.loc[mask_negative, 'clean_tokens'].to_list())



top_10_positive = FreqDist(all_positive_tokens).most_common(10)

top_10_negative = FreqDist(all_negative_tokens).most_common(10)
top_10_positive
top_10_negative
def make_dict(clean_tokens):

    return dict([token, True] for token in clean_tokens)
df['dict'] = df['clean_tokens'].apply(make_dict)



df.head()
df['tuple'] = df.apply(lambda row : (row['dict'], row['sentiment']), axis=1)



df.head()
df.shape
train = df.loc[:7000, 'tuple']

test = df.loc[7000:, 'tuple']
classifier = NaiveBayesClassifier.train(train)

acc = classify.accuracy(classifier, test)



print("The accuracy of the classifier is {:.2f}%.".format(acc))

print(classifier.show_most_informative_features(10))
filename = 'twitter_model.sav'

pickle.dump(classifier, open(filename, 'wb'))