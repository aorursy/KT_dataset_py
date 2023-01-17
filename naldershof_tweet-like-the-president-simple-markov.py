import numpy as np

import pandas as pd
tweets = pd.read_csv('../input/Donald-Tweets!.csv')
print("There are a total of {} tweets in the dataset".format(len(tweets)))
tweets['clean_text'] = tweets.Tweet_Text.str.replace(r'http.*(\s|$)','<link>')
def create_structures(corpus):

    # So we'll construct a dictionary that looks like

    # {'word':[a,a,a,b]}

    # Then when we construct a sentence we'll look at the key and pick

    # a random thing from it's values



    markov_dict = {'<EOT>':[]}



    # Also keep a list of starting_words to easily choose a starting word

    # based on how often they show up

    # Note: we're doing this in a sorta convoluted way, but we're already iterating

    # over all the words, no reason not to do that here while we're at it.



    starting_words = []



    for tweet in corpus.clean_text:

        tok_tweet = tweet.split()

        word_count = len(tok_tweet)

        for index, word in enumerate(tok_tweet):

            if word not in markov_dict.keys():

                markov_dict[word] = []



            if index == word_count - 1:

                markov_dict[word].append("<EOT>")

                #couplet = (tok_tweet[index], "<EOT>")

            else:

                if index == 0:

                    starting_words.append(word)

                markov_dict[word].append(tok_tweet[index+1])

    return markov_dict, starting_words
markov_dict, starting_words = create_structures(tweets)
def write_tweet(starting_word, chain):

    tweet = starting_word

    current_word = starting_word

    

    while len(tweet) <= 140:        

        next_word = np.random.choice(chain[current_word])

        if next_word == '<EOT>':

            return tweet

        

        new_tweet = tweet + ' ' + next_word

        if  len(new_tweet) > 140:

            return tweet

        else:

            tweet = new_tweet

            current_word = next_word



for x in range(0,15):

    starting_word = np.random.choice(starting_words)

    print(write_tweet(starting_word, markov_dict)+"\n")
def simplified_structures(corpus):

    markov_dict = {'<EOT>':[]}

    starting_words = []



    for tweet in corpus.clean_text:

        tweet = tweet.upper()

        tweet = tweet.replace(",","")

        tweet = tweet.replace("\"","")

        tok_tweet = tweet.split()

        word_count = len(tok_tweet)

        for index, word in enumerate(tok_tweet):

            if word not in markov_dict.keys():

                markov_dict[word] = []



            if index == word_count - 1:

                markov_dict[word].append("<EOT>")

            else:

                if index == 0:

                    starting_words.append(word)

                markov_dict[word].append(tok_tweet[index+1])

    return markov_dict, starting_words
simple_markov, simple_start = simplified_structures(tweets)
print("Original structure has {} total results".format(len(markov_dict)))

print("Simplified structure has {} total results".format(len(simple_markov)))



simplification = np.round(100 * len(simple_markov) / len(markov_dict))

print("We've reduced the dataset to {}% the original size".format(simplification))
for x in range(0,15):

    starting_word = np.random.choice(simple_start)

    print(write_tweet(starting_word, simple_markov)+"\n")
print("Options to follow FACE:")

print(set(simple_markov['FACE']))

print("\nNumber of Options to follow THE:")

print(len(set(simple_markov['THE'])))

print("\nOptions to follow MEDIA:")

print(set(simple_markov['MEDIA']))

print("\nOptions to follow CREATION.:")

print(set(simple_markov['CREATION.']))