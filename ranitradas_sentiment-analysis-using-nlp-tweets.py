# Pre defining the functions - 

# 1) process_tweet - for pre-processing the tweets

# 2) build_freqs - for creating frequency dictionary



import re

import string

import numpy as np



from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer





def process_tweet(tweet):

    """Process tweet function.

    Input:

        tweet: a string containing a tweet

    Output:

        tweets_clean: a list of words containing the processed tweet



    """

    stemmer = PorterStemmer()

    

    stopwords_english = stopwords.words('english')

    

    # remove stock market tickers like $GE

    tweet = re.sub(r'\$\w*', '', tweet)

    

    # remove old style retweet text "RT"

    tweet = re.sub(r'^RT[\s]+', '', tweet)

    

    # remove hyperlinks

    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    

    # remove hashtags

    # only removing the hash # sign from the word

    tweet = re.sub(r'#', '', tweet)

    

    # tokenize tweets

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,

                               reduce_len=True)

    tweet_tokens = tokenizer.tokenize(tweet)



    tweets_clean = []

    for word in tweet_tokens:

        if (word not in stopwords_english and  # remove stopwords

                word not in string.punctuation):  # remove punctuation

            # tweets_clean.append(word)

            stem_word = stemmer.stem(word)  # stemming word

            tweets_clean.append(stem_word)



    return tweets_clean





def build_freqs(tweets, ys):

    """Build frequencies.

    Input:

        tweets: a list of tweets

        ys: an m x 1 array with the sentiment label of each tweet

            (either 0 or 1)

    Output:

        freqs: a dictionary mapping each (word, sentiment) pair to its

        frequency

    """

    # The squeeze is necessary or the list ends up with one element

    yslist = np.squeeze(ys).tolist()



    # Start with an empty dictionary and populate it by looping over all tweets

    # and over all processed words in each tweet.

    freqs = {}

    for y, tweet in zip(yslist, tweets):

        for word in process_tweet(tweet):

            pair = (word, y)

            if pair in freqs:

                freqs[pair] += 1

            else:

                freqs[pair] = 1



    return freqs
# Importing libraries and modules

import nltk

import numpy as np

import pandas as pd

from nltk.corpus import twitter_samples
# select the set of positive and negative tweets

all_positive_tweets = twitter_samples.strings('positive_tweets.json')

all_negative_tweets = twitter_samples.strings('negative_tweets.json')
# split the data into two pieces, one for training and one for testing (validation set) 

test_pos = all_positive_tweets[4000:]

train_pos = all_positive_tweets[:4000]

test_neg = all_negative_tweets[4000:]

train_neg = all_negative_tweets[:4000]



train_x = train_pos + train_neg 

test_x = test_pos + test_neg
# combine positive and negative labels

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)

test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
#Print the shape of train and test sets

print("train_y.shape =" + str(train_y.shape))

print("test_y.shape =" + str(test_y.shape))
#Create Frequency Distribution

freqs = build_freqs(train_x,train_y)



#Check the output

print("type(freqs) = " + str(type(freqs)))

print("len(freqs) = " + str(len(freqs.keys())))
#Processing



#Test the function below

print('This is an example of a positive tweet: \n', train_x[0])

print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))
#Sigmoid function

def sigmoid(z):

    h = 1/(1+np.exp(-z))

    return h
# Testing the sigmoid function 

if (sigmoid(0) == 0.5):

    print('SUCCESS!')

else:

    print('Oops!')



if (sigmoid(4.92) == 0.9927537604041685):

    print('CORRECT!')

else:

    print('Oops again!')
#Gradient Descent

def gradientDescent(x, y, theta, alpha, num_iters):

    m = x.shape[0]

    for i in range(0, num_iters):

        z = np.dot(x,theta)

        h = sigmoid(z)

        J = -1./m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(),np.log(1-h)))

        theta = theta = theta - (alpha/m) * np.dot(x.transpose(),(h-y))

    J = float(J)

    return J, theta
# Check the function

# Construct a synthetic test case using numpy PRNG functions

np.random.seed(1)

# X input is 10 x 3 with ones for the bias terms

tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)

# Y Labels are 10 x 1

tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)



# Apply gradient descent

tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)

print(f"The cost after training is {tmp_J:.8f}.")

print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")
#Extracting Features

def extract_features(tweet, freqs):

    word_l = process_tweet(tweet)

    x = np.zeros((1, 3))

    x[0,0] = 1

    for word in word_l:

        x[0,1] += freqs.get((word, 1.0),0)

        x[0,2] += freqs.get((word, 0.0),0)

    assert(x.shape == (1, 3))

    return x
# Checking the function



# test 1:

# test on training data

tmp1 = extract_features(train_x[0], freqs)

print(tmp1)





# test 2:

# check for when the words are not in the freqs dictionary

tmp2 = extract_features('blorb bleeeeb bloooob', freqs)

print(tmp2)
#Training the Model



# Collect the features 'x' and stack them into a matrix 'X'

X = np.zeros((len(train_x), 3))

for i in range(len(train_x)):

    X[i, :]= extract_features(train_x[i], freqs)



# Training labels corresponding to X

Y = train_y



# Apply gradient descent

J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

print(f"The cost after training is {J:.8f}.")

print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
# Prediction



def predict_tweet(tweet, freqs, theta):

    # extract the features of the tweet and store it into x

    x = extract_features(tweet,freqs)

    

    # make the prediction using x and theta

    y_pred = sigmoid(np.dot(x,theta))

    return y_pred
# Testing the function

for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:

    print( '%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))
# Check performance using the test set

def test_logistic_regression(test_x, test_y, freqs, theta):

    y_hat = []

    

    for tweet in test_x:

        # get the label prediction for the tweet

        y_pred = predict_tweet(tweet, freqs, theta)

        

        if y_pred > 0.5:

            y_hat.append(1)

        else:

            y_hat.append(0)



    # With the above implementation, y_hat is a list, but test_y is (m,1) array

    # convert both to one-dimensional arrays in order to compare them using the '==' operator

    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)

    return accuracy
tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)

print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")
# Error Analysis

print('Label Predicted Tweet')

for x,y in zip(test_x,test_y):

    y_hat = predict_tweet(x, freqs, theta)



    if np.abs(y - (y_hat > 0.5)) > 0:

        print('THE TWEET IS:', x)

        print('THE PROCESSED TWEET IS:', process_tweet(x))

        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))
#Predict your own tweet



my_tweet = 'I have completed the coding part. I am very happy.'

print(process_tweet(my_tweet))

y_hat = predict_tweet(my_tweet, freqs, theta)

print(y_hat)

if y_hat > 0.5:

    print('Positive sentiment')

else: 

    print('Negative sentiment')