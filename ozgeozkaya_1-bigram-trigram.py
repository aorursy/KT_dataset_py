# This is the first homework of NLP course

# The aim of the homework is finding bigrams and trigrams from a article series

# The text taken from "BBC articles fulltext and category" and includes only business category

# The text has been cleared from stopwords
import nltk

import pandas as pd

from nltk import ngrams

import numpy as np

import matplotlib.pyplot as plt



text = open('../input/filtered_business.txt', 'r').read()  # Open given text
# Function n_gram: splits the words in order to given parameter

# Parameter n: it's the "n" number for "n-gram"

def n_gram(n):                                             

    n_grams = ngrams(text.split(), n)

    return n_grams
sentences = []

df_frequency = []
# Function frequency: counts the frequency of the each "n-gram" output

# Parameter grammed: it takes the function "n_gram"s return as value. Briefly, the grammed text.

def frequency(grammed):

    sentences.clear()

    df_frequency.clear()

    freq = nltk.FreqDist(grammed)

    for k, v in freq.items():

        sentences.append(k)                     # Sentences is a list, stores the grams(ignores duplicates)

        df_frequency.append(v)                  # df_frequency is a list, stores the frequency of grams
# Function create_plot: it creates a plot for given grams

# Parameter num: it's a number to send it to "n_gram" function

def create_plot(num):

    frequency(n_gram(num)) # Send "num" parameter to "n_gram func." and send the result to "frequency func."



    gram_frame = pd.DataFrame(sentences)       # gram_frame is the data frame to store grams and freq.



    gram_frame['frequencies'] = df_frequency

    if num == 2:

        gram_frame.columns = ['first', 'second', 'frequencies']

    if num == 3:

        gram_frame.columns = ['first', 'second', 'third', 'frequencies']



    gram_frame.sort_values("frequencies", axis=0, ascending=False, inplace=True, na_position='last')



    gram_frame = gram_frame.head(20)            # Only take the top 20 of gram_frame



    total = sum(df_frequency)



    gram_frame["ratio"] = gram_frame['frequencies'].div(total)   # Additional, ratio is added



    plt.rcdefaults()

    fig, ax = plt.subplots()



    if num == 2:

        grams = gram_frame["first"] + " " + gram_frame["second"]

    if num == 3:

        grams = gram_frame["first"] + " " + gram_frame["second"] + " " + gram_frame["third"]



    # Create plot

    y_pos = np.arange(len(grams))

    performance = gram_frame["frequencies"]



    ax.barh(y_pos, performance)

    ax.set_yticks(y_pos)

    ax.set_yticklabels(grams)

    ax.invert_yaxis()  # labels read top-to-bottom

    ax.set_xlabel('Frequency')

    ax.set_title('2-3 grams')



    plt.show()

    display(gram_frame)
create_plot(2)   # Create plot for bigram 
create_plot(3)   # Create plot for trigram