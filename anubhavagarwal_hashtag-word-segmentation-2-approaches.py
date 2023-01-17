# This Kaggle - Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/tweet_data/tweet_data"))

print(os.listdir("../input/tweet_data/tweet_data/word_dict_ngram"))



# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

"""

Created on Thu Jun 19 11:59:34 2019



@author: AnubhavA

"""



"""English Word Segmentation using unigram and bigram data in Python



Reference:

    https://github.com/grantjenks/python-wordsegment



Source of unigram and bigram files:

    http://norvig.com/ngrams/ under the names count_1w.txt and count_2w.txt

 

"""



import io

import math





dir_path = "../input/tweet_data/tweet_data/word_dict_ngram"



class Segmenter(object):



    ALPHABET = set('abcdefghijklmnopqrstuvwxyz0123456789')  

    WORDS_FILENAME = dir_path + "/words.txt"

    BIGRAMS_FILENAME = dir_path + "/bigrams.txt"

    UNIGRAMS_FILENAME = dir_path +"/unigrams.txt"

    TOTAL = 1024908267229.0 #is the total number of words in the corpus ##Natural Language Corpus Data: Beautiful Data

    LIMIT = 24

   

    

    def __init__(self):

        "Initialize the class variables"

        self.unigrams = {}

        self.bigrams = {}

        self.total = 0.0

        self.limit = 0

        self.words = []

    

    @staticmethod

    def parse(filename):

        "Read `filename` and parse tab-separated file of word and count pairs."

        with io.open(filename, encoding='utf-8') as reader:

            lines = (line.split('\t') for line in reader)

            return dict((word, float(number)) for word, number in lines)

        



    def load(self):

        "Load unigram and bigram counts from local disk storage."

        self.unigrams.update(self.parse(self.UNIGRAMS_FILENAME))

        self.bigrams.update(self.parse(self.BIGRAMS_FILENAME))

        self.total = self.TOTAL

        self.limit = self.LIMIT

        with io.open(self.WORDS_FILENAME, encoding='utf-8') as reader:

            text = reader.read()

            self.words.extend(text.splitlines())





    def score(self, word, previous=None):

        "Score each `word` in the context of `previous` word."

        unigrams = self.unigrams

        bigrams = self.bigrams

        total = self.total



        if previous is None:

            if word in unigrams:



                # Probability of the given word.

                return unigrams[word] / total



            # Penalize words not found in the unigrams according

            # to their length

            return 10.0 / (total * 10 ** len(word))



        bigram = '{0} {1}'.format(previous, word)



        if bigram in bigrams and previous in unigrams:



            # Conditional probability of the word given the previous

            # word. 

            return bigrams[bigram] / total / self.score(previous)



        # Fall back to using the unigram probability.

        return self.score(word)





    def isegment(self, text):

        "Return iterator of words that is the best segmenation of `text`."

        memo = dict()



        def search(text, previous='<s>'):

            "Return max of candidates matching `text` given `previous` word."

            if text == '':

                return 0.0, []



            def candidates():

                "Generator of (score, words) pairs for all divisions of text."

                for prefix, suffix in self.divide(text):

                    prefix_score = math.log10(self.score(prefix, previous))



                    pair = (suffix, prefix)

                    if pair not in memo:

                        memo[pair] = search(suffix, prefix)

                    suffix_score, suffix_words = memo[pair]



                    yield (prefix_score + suffix_score, [prefix] + suffix_words)



            return max(candidates())



        # Avoid recursion limit issues by dividing text into chunks, segmenting

        # those chunks and combining the results together.



        clean_text = self.clean(text)

        size = 250

        prefix = ''



        for offset in range(0, len(clean_text), size):

            chunk = clean_text[offset:(offset + size)]

            _, chunk_words = search(prefix + chunk)

            prefix = ''.join(chunk_words[-5:])

            del chunk_words[-5:]

            for word in chunk_words:

                yield word



        _, prefix_words = search(prefix)



        for word in prefix_words:

            yield word





    def segment(self, text):

        "Return list of words that is the best segmenation of input `text`."

        return list(self.isegment(text))





    def divide(self, text):

        "Yield `(prefix, suffix)` pairs from input `text`."

        for pos in range(1, min(len(text), self.limit) + 1):

            yield (text[:pos], text[pos:])





    @classmethod

    def clean(cls, text):

        "Return `text` lower-cased with non-alphanumeric characters removed."

        alphabet = cls.ALPHABET

        text_lower = text.lower()

        letters = (letter for letter in text_lower if letter in alphabet)

        return ''.join(letters)







if __name__ == '__main__':

    print("Enter tweet containing #hashtag: ")

    

    ## uncomment to get user input

    #input_hashtag = input()

    

    ## comment if you enable user input

    input_hashtag = "#goldenglobes2015  #movieaddict"

    

    segmenter = Segmenter()  

    load = segmenter.load()

    clean = segmenter.clean(input_hashtag) 

    isegment = segmenter.isegment(clean) 

    

    print("Input: {0}".format(input_hashtag))

    print("Segmented word list: {0}".format(list(isegment)))

    





# -*- coding: utf-8 -*-

"""

Created on Sun Jun 18 10:37:46 2019



@author: AnubhavA



Finding Maximal Munch in Python



Source of dictionery:

    https://github.com/first20hours/google-10000-english

"""



#!/usr/bin/env python

# -*- coding: utf-8 -*-

from collections import defaultdict

import re

import string



dir_path = "../input/tweet_data/tweet_data/"



with open(dir_path + 'google-10000-english.txt') as reader:

    words = [line.split('\n')[0] for line in reader]



words.sort()



dictionary =  list(set(words))



swap_dict = { "im":"i'm","its":"it's","thats":"that's","ill":"i'll","ure":"u're","hes":"he's","hesnt":"hesn't","shes":"she's","shesnt":"shesn't","shant": "shan't",

        "theyre":"they're","theyll":"they'll","cant":"can't","dont":"don't","wont":"won't","arent":"aren't","wouldnt":"wouldn't","shouldnt":"shouldn't",

        "couldnt":"couldn't","havent":"haven't","hasnt":"hasn't","oclock":"o'clock","iam":"i am", "youare":"you are","heis":"he is","sheis":"she is", 

        "arent": "aren't","couldve": "could've","didnt": "didn't","doesnt": "doesn't","dont": "don't",

        "hadnt": "hadn't","hed": "he'd","hell": "he'll","hes": "he's","howd": "how'd","howll": "how'll","hows": "how's","id": "i'd","ive": "i've",

        "isnt": "isn't","lets": "let's","mustnt": "mustn't","mustve": "must've","shed": "she'd","shell": "she'll","shouldve": "should've",

        "somebodys": "somebody's","someones": "someone's","somethings": "something's","thats": "that's","thatll": "that'll","theres": "there's",

        "theyve": "they've","wasnt": "wasn't","were": "we're","weve": "we've","werent": "weren't","whatll": "what'll","whats": "what's","whatve": "what've",

        "wholl": "who'll","whos": "who's","whove": "who've","youll": "you'll","youre": "you're","youve": "you've"

       }

   

def maximal_munch(hashtag):

    i = 0  # starting position

    tokens = []

    tries = defaultdict(list)

    moves_examined = [False] * len(hashtag)



    while i < len(hashtag):



        # Generate possible moves from current step

        if not moves_examined[i]:

            j = i  # ending position

            while j < len(hashtag):

                if hashtag[i:j+1] in dictionary:

                    tries[i].append(j)

                j += 1

            moves_examined[i] = True



        # Pick "maximal munch" from current move if possible

        # If not possible, revert

        if len(tries[i]) > 0:

            maximum_token_j_position = tries[i].pop()

            tokens.append(hashtag[i:maximum_token_j_position+1])

            i = maximum_token_j_position + 1

        else:

            if i == 0:

                return ''



            failed_token = tokens.pop()

            i -= len(failed_token)



    if i == len(hashtag):

        return ' '.join(tokens)



def post_processor(s):

    for i in swap_dict:

        s = s.replace(i, swap_dict[i])

    return s



end_s = re.compile(r' s$')



def post_processor_2(s):

    s = s.replace('e r', "er")

    s = s.replace(' s ', 's ')

    s = end_s.sub('s', s)

    return s





if __name__ == '__main__':

    output = []

    print("Enter twitter hashtag:")

    ## uncomment to get user input

    #input_hashtag = input()

    

    ## comment if you enable user input

    input_hashtag = "#GoldenGlobes #AmericanSniper"

    

    

    for part in input_hashtag.split():

        if part.startswith('#'):            

            process_input = post_processor(maximal_munch(part[1:].strip().lower()))

            output.append(post_processor_2(process_input))            

        else:

            process_input = post_processor(maximal_munch(part.strip().lower()))

            output.append(post_processor_2(process_input))

    

    print("Input: {0}".format(input_hashtag))

    print("Segmented word list: {0}".format(output)) 

    



def extract_hash_tags(tweet):

    "extract hashtags from the tweet"

    return set(part[1:] for part in tweet.split() if part.startswith('#'))



import string

import pandas as pd



hashtags =[]



punct = string.punctuation

exclude_punct = set(punct)

exclude_punct_no_hash = set(punct.replace('#',''))



## read data from tweet corpus

twt_file = open("../input/tweet_data/tweet_data/tweets.txt", "r", encoding="utf8")

for twt in twt_file: 

    "clean a tweet and extract hashtags in it"

    non_ASCII_free = (twt.encode('ascii', 'ignore')).decode("utf-8")

    punc_free = ''.join([ch for ch in non_ASCII_free if ch not in exclude_punct_no_hash])

    short_word_free = ' '.join(word for word in punc_free.split() )

    tags = extract_hash_tags(short_word_free)

    for tag in tags:

        if(len(tag)>0):

            hashtags.append(tag.lower())



##unique hashtags        

hashtags = list(set(hashtags))



print("Total unique hashtags found: {0}".format(len(hashtags)))



print("processing...")



hashtag_columns = ['hashtag', 'word_segment']

data_frame = pd.DataFrame(columns=hashtag_columns)



segmenter = Segmenter()  

load = segmenter.load()

    

    

for hashtag in hashtags:

    clean = segmenter.clean(hashtag) 

    isegment = segmenter.isegment(clean) 

    word_segment= list(isegment)    

    data_frame = data_frame.append({'hashtag': "#" + hashtag, 'word_segment': word_segment}, ignore_index=True)

    





##print few hashtag and word segmented

print("List of few hashtags")

data_frame.head(10)