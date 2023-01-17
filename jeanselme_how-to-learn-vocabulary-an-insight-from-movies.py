# Import all useful libraries

## For math

import numpy as np

import pandas as pd

from scipy import optimize

from scipy.stats import median_absolute_deviation



## For visualization

import seaborn as sns

import matplotlib.pyplot as plt



## For text

from keras.preprocessing.text import Tokenizer

from nltk.stem import WordNetLemmatizer
# Open conversation data

conversation = pd.read_csv('../input/movie-dialog-corpus/movie_lines.tsv', encoding='utf-8-sig', header=None) ## Open file

conversation = conversation[0].str.split('\t') ## Split given t (cannot be done directly in read_csv because text format)

conversation = conversation.apply(lambda x: x[4]) ## Select only text
# Tokenize text, ie extract words

keras_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}\t\n1234567890')

keras_tokenizer.fit_on_texts(conversation)
# How many words are used in movies ?

print("This dataset contains {} words".format(len(keras_tokenizer.word_counts)))
# Sort words by count

word_counts = pd.Series(keras_tokenizer.word_counts)

word_counts = word_counts[word_counts.index.to_series().apply(len) > 2] # Remove one and two letter words

word_frequency = word_counts / word_counts.sum()
# Find roots of each words

lemmatizer = WordNetLemmatizer()

roots = word_frequency.index.to_series().apply(lemmatizer.lemmatize)
# Merge counts of same root words

word_frequency = word_frequency.groupby(roots.values).sum().sort_values()[::-1]

word_counts = word_counts.groupby(roots.values).sum()

print("It remains {} words".format(len(word_frequency)))
percentages = np.linspace(0.1, 1, 10)
first, length = {}, {}

for percentage in percentages:

    # Select words until the given percentage 

    selection = word_frequency[word_frequency.cumsum() < percentage].index

    

    # Extract lengths and first letters

    length_words = selection.to_series().apply(len)

    first_letter = selection.to_series().apply(lambda x: x[0]).value_counts()

    first_letter /= first_letter.sum()

    

    first[percentage] = first_letter[:3]

    length[percentage] = {"median": length_words.mean(), "confidence": median_absolute_deviation(length_words)}
ax = pd.DataFrame(first).T.plot.bar(stacked = True)

ax.legend(title = 'First letter', loc='center left', bbox_to_anchor=(1, 0.5))

ax.set_xlabel('Percentage of vocabulary learnt')

ax.set_ylabel('Percentage words starting with letter')
length = pd.DataFrame(length).T

ax = length['median'].plot()

ax.fill_between(length.index, length['median'] - length.confidence, length['median'] + length.confidence, alpha = 0.2)

ax.set_xlabel('Percentage of vocabulary learnt')

ax.set_ylabel('Median length word')
ax = word_frequency.cumsum().plot()

ax.set_ylabel('Percentage voca covered')

ax.set_xlabel('Vocabulary (from common to less )')
# Fit on words that are a minimum used to avoid fitting long tail

fit_on = word_frequency[word_counts > 50].cumsum()
def pareto(x, alpha):

    # CDF of a pareto distribution

    return 1 - (1. / x) ** alpha
# Optimization

x = np.arange(1, len(fit_on) + 1)

alpha = optimize.curve_fit(pareto, x, fit_on)[0][0]
ax = fit_on.plot()

ax.set_ylabel('Percentage voca covered')

ax.set_xlabel('Vocabulary')

ax.plot(x, pareto(x, alpha))
def error(percentage_words):

    # Absolute error 

    num_words = int(percentage_words * len(word_frequency))

    return np.abs(percentage_words - (1 - word_frequency[:num_words].sum()))
# Optimization

optimize.minimize(error, 0.1, bounds = [(0, 1)], method = 'L-BFGS-B', tol = 10**-3)
percentage_understood = 0.90 # Percentage that I would understand of movies
selection = word_frequency[word_frequency.cumsum() < percentage_understood].index

selection
# Start your training by running this loop !

train = False # Put True if you wanna train 



boxes = np.ones(len(selection))

num_boxes = 5

while train and boxes.min() != num_boxes:

    # Compute probabilities

    probabilities = num_boxes - boxes # Far boxes are less likely    

    probabilities /= np.sum(probabilities) # Normalize porbability

    

    # Draw a word

    i = np.random.choice(len(selection), p = probabilities)

    

    # Ask if you know word 

    print("Do you know the word: {} ? (y/n | s to stop)".format(selection[i]), end = ' ')

    answer = input()

    

    if answer == 's':

        break



    # Update probability

    known = 2 * (answer == 'y') - 1 # +1 if known -1 if not

    probabilities[i] = max(min(num_boxes, probabilities[i] + known), 1) # If you know go up, if you don't go down