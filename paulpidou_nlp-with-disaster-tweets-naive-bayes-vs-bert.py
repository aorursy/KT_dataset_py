import numpy as np

import pandas as pd

import string

import re

from matplotlib import pyplot as plt



import nltk

from nltk.tokenize import TweetTokenizer

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Let's use the Natural Language ToolKit: https://www.nltk.org/

nltk.download('stopwords')
train_filepath = '/kaggle/input/nlp-getting-started/train.csv'

df_train = pd.read_csv(train_filepath, index_col='id')

df_train.head()
# Percentage of missing values

df_train.isnull().sum() / len(df_train) * 100
disaster_ratio = df_train.groupby('target').count()

disaster_ratio.plot(kind='pie', y='text', labels=['Not Disaster', 'Disaster'])



not_disater = disaster_ratio.iloc[0]['text'] / len(df_train) * 100

print("Disaster :", 100-not_disater, "%")

print("Not Disaster :", not_disater, "%")
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

stopwords_english = stopwords.words('english') 

stemmer = PorterStemmer()
def process_tweet(tweet, tokenizer=tokenizer, stopwords_english=stopwords_english, stemmer=stemmer):

    tweet = re.sub(r'^RT[\s]+', '', tweet)  # remove old style retweet text "RT"

    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)  # remove hyperlinks

    tweet = re.sub(r'#', '', tweet)  # remove hashtags



    tweet_tokens = tokenizer.tokenize(tweet)

    

    tweets_clean = []

    for word in tweet_tokens:

        if (word not in stopwords_english and word not in string.punctuation):

            tweets_clean.append(word)

            

    tweets_stem = [] 

    for word in tweets_clean:

        stem_word = stemmer.stem(word)

        tweets_stem.append(stem_word)

        

    return tweets_stem
def count_tweets(result, tweets, ys):

    '''

    Input:

        result: a dictionary that will be used to map each pair to its frequency

        tweets: a list of tweets

        ys: a list indicating if each tweet is a disaster or not (either 0 or 1)

    Output:

        result: a dictionary mapping each pair to its frequency

    '''



    for y, tweet in zip(ys, tweets):

        for word in process_tweet(tweet):

            pair = (word, y)

            result[pair] = result.get(pair, 0) + 1



    return result
def train_naive_bayes(freqs, train_x, train_y):

    '''

    Input:

        freqs: dictionary from (word, label) to how often the word appears

        train_x: a list of tweets

        train_y: a list of labels correponding to the tweets (0,1)

    Output:

        logprior: the log prior

        loglikelihood: the log likelihood of you Naive bayes equation

    '''

    loglikelihood = {}

    logprior = 0



    vocab = set([pair[0] for pair in freqs.keys()])

    V = len(vocab)



    N_ndis = N_dis = 0

    for pair in freqs.keys():

        if pair[1] > 0:

            N_dis += freqs.get(pair, 0)

        else:

            N_ndis += freqs.get(pair, 0)



    D = len(train_x)

    D_dis = np.sum(train_y)

    D_ndis = D - D_dis



    logprior = np.log(D_dis) - np.log(D_ndis)



    for word in vocab:

        freq_ndis = freqs.get((word, 0), 0)

        freq_dis = freqs.get((word, 1), 0)



        p_w_ndis = (freq_ndis + 1) / (N_ndis + V)

        p_w_dis = (freq_dis + 1) / (N_dis + V)



        loglikelihood[word] = np.log(p_w_dis / p_w_ndis)



    return logprior, loglikelihood
def naive_bayes_predict(tweet, logprior, loglikelihood):

    '''

    Input:

        tweet: a string

        logprior: a number

        loglikelihood: a dictionary of words mapping to numbers

    Output:

        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)



    '''

    word_l = process_tweet(tweet)



    p = 0

    p += logprior



    for word in word_l:

        if word in loglikelihood:

            p += loglikelihood.get(word, 0)



    return p
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):

    """

    Input:

        test_x: A list of tweets

        test_y: the corresponding labels for the list of tweets

        logprior: the logprior

        loglikelihood: a dictionary with the loglikelihoods for each word

    Output:

        accuracy: (# of tweets classified correctly)/(total # of tweets)

    """

    accuracy = 0



    y_hats = []

    for tweet in test_x:

        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:

            y_hat_i = 1

        else:

            y_hat_i = 0



        y_hats.append(y_hat_i)



    error = np.mean(np.abs(y_hats - test_y))

    accuracy = 1 - error



    return accuracy
freqs = count_tweets({}, df_train['text'], df_train['target'])

logprior, loglikelihood = train_naive_bayes(freqs, df_train['text'], df_train['target'])



print(logprior)

print(len(loglikelihood))
print("Naive Bayes accuracy:", test_naive_bayes(df_train['text'], df_train['target'], logprior, loglikelihood) * 100, "%")
print('Truth Predicted Tweet')

for x, y in zip(df_train['text'], df_train['target']):

    y_hat = naive_bayes_predict(x, logprior, loglikelihood)

    if y != (np.sign(y_hat) > 0):

        print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(

            process_tweet(x)).encode('ascii', 'ignore')))
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_test['target'] = df_test['text'].apply(lambda tweet : int(np.sign(naive_bayes_predict(tweet, logprior, loglikelihood)) > 0))
submission = df_test[['id', 'target']]

submission.to_csv('nb_submission.csv', index=False)  # Public score 0.78976
# Official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import tokenization
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# Remove target column from df_test

if 'target' in df_test.columns.tolist():

    del df_test['target']
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
train_input = bert_encode(df_train.text.values, tokenizer, max_len=160)

test_input = bert_encode(df_test.text.values, tokenizer, max_len=160)

train_labels = df_train.target.values
def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
model = build_model(bert_layer, max_len=160)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=2,

    callbacks=[checkpoint],

    batch_size=16

)
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train'], loc='upper left')

plt.show()
model.load_weights('model.h5')

test_pred = model.predict(test_input)
df_test['target'] = test_pred.round().astype(int)



submission = df_test[['id', 'target']]

submission.to_csv('bert_submission.csv', index=False)