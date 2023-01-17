import nltk

import pandas as pd

import numpy as np

import re



# If you would like to work with the raw text you can use 'moby_raw'

with open('../input/moby.txt', 'r') as f:

    moby_raw = f.read()

    

# If you would like to work with the novel in nltk.Text format you can use 'text1'

moby_tokens = nltk.word_tokenize(moby_raw)

text1 = nltk.Text(moby_tokens)
def example_one():

    

    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)



example_one()
def example_two():

    

    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))



example_two()
from nltk.stem import WordNetLemmatizer



def example_three():



    lemmatizer = WordNetLemmatizer()

    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]



    return len(set(lemmatized))



example_three()
def answer_one():

    

    

    return example_two() / example_one()



answer_one()
def answer_two():

    tokens = nltk.word_tokenize(moby_raw)

    whales = [w for w in tokens if  w =='whale' or w == 'Whale'] 

    return len(whales) / example_one()



answer_two()
def answer_three():

    tokens = nltk.word_tokenize(moby_raw)

    dist = nltk.FreqDist(tokens)



    return dist.most_common(20)



answer_three()
def answer_four():

    tokens = nltk.word_tokenize(moby_raw)

    dist = nltk.FreqDist(tokens)

    vocab1 = list(dist.keys())

    sel_list = [t for t in vocab1 if len(t)>5 and dist[t]>150]

    return sorted(sel_list)



answer_four()
def answer_five():

    tokens = nltk.word_tokenize(moby_raw)

    tokens_uni = set(tokens)

    longest_word = max(tokens_uni, key=len)

    result = longest_word, len(longest_word)

    

    return result



answer_five()
def answer_six():

    tokens = nltk.word_tokenize(moby_raw)

    dist = nltk.FreqDist(tokens)

    freq_words = [w for w in list(dist.keys()) if w.isalpha() and dist[w] > 2000]

    

    dict_I_want = { key: dist[key] for key in freq_words } # make new dict

    result = sorted(dict_I_want.items(), key=lambda kv: kv[1], reverse=True)

    # read more about *sorted* function



    return result

    

answer_six()
def answer_seven():

    # count tokens

    tokens = len(nltk.word_tokenize(moby_raw))

    # count sentences

    sents = len(nltk.sent_tokenize(moby_raw))

    # divide # of tokens by # of sentences

    

    return tokens/sents



answer_seven()
def answer_eight():

    tokens = nltk.word_tokenize(moby_raw)

    tags = nltk.pos_tag(tokens)

    frequencies = nltk.FreqDist([tag for (word, tag) in tags])

    return frequencies.most_common(5)



answer_eight()
from nltk.corpus import words

from nltk.metrics.distance import jaccard_distance, edit_distance

from nltk.util import ngrams



correct_spellings = words.words()
entries=['cormulent', 'incendenece', 'validrate']
def jaccard(entries, gram_number):

    """find the closet words to each entry



    Args:

     entries: collection of words to match

     gram_number: number of n-grams to use



    Returns:

     list: words with the closest jaccard distance to entries

    """

    outcomes = []

    for entry in entries:

        spellings = [s for s in correct_spellings if s.startswith(entry[0])]

        distances = ((jaccard_distance(set(ngrams(entry, gram_number)),

                                       set(ngrams(word, gram_number))), word) for word in spellings)

        closest = min(distances)

        outcomes.append(closest[1])

    return outcomes



jaccard(entries, 3)
jaccard(entries, 4)
def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):

    """gets the nearest words based on Levenshtein distance



    Args:

     entries (list[str]): words to find closest words to



    Returns:

     list[str]: nearest words to the entries

    """

    outcomes = []

    for entry in entries:

        distances = ((edit_distance(entry,

                                    word), word)

                     for word in correct_spellings)

        closest = min(distances)

        outcomes.append(closest[1])

    return outcomes



print(answer_eleven())
