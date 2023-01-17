# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from numpy import argmax
from re import match
from nltk.tokenize import word_tokenize
from random import choice
from fuzzywuzzy import process
# =============================================================================
# load and filter data
# =============================================================================
data = pd.read_csv('../input/winemag-data-130k-v2.csv')
data = data.filter(items=['country','variety','description'])

# Further filter for country and variety such that only those with greater than 1000 samples remain
df_dummy = data.groupby('country').filter(lambda x: len(x) > 1000)
filteredData = df_dummy.groupby('variety').filter(lambda x: len(x) > 1000)

# Output lists of countries and varieties with sufficient number of samples in original data
print("Countries you can choose from include: \n{}".format(list(filteredData.country.unique())))
print("\nVarieties you can choose from include: \n{}".format(list(filteredData.variety.unique())))

# =============================================================================
# To correct for inaccurate spelling of certain wine varieties, function 
# `mineText' determines variety and country to be that most closely matching 
# your input - using function edit_distance. E.g.
# variety = 'temparmillo'
# trueVariety = 'Tempranillo'
# =============================================================================
  
def mineText(df, country, variety):
    N = 1000 # a minimum of 1000 reviews to sample from
    # trueVariety and trueCountry incase of incorrect spelling input
    trueVariety = process.extractOne(variety, filteredData.variety.unique())[0]
    trueCountry = process.extractOne(country, filteredData.country.unique())[0]
    
    print("You're drinking a {} from {}".format(trueVariety, trueCountry))
    
    # Reviews from samples having matching country AND variety
    minedText = df[ (df["country"] == trueCountry) & (df["variety"] == trueVariety)]['description']
    
    # if too few samples, take samples having either matching country OR variety
    # concatenate this to previous sampled reviews
    if len(minedText) < N:
        # bitwise exclusive OR ^
        moreText = df[ (df["country"] == trueCountry) ^ (df["variety"] == trueVariety)]['description']
        minedText = pd.concat([minedText, moreText.sample(n = N - len(minedText))], axis=0)      
    # return reviews concatenated into one single string
    return minedText.str.cat(sep=' ')

minedText = mineText(filteredData, 'franc', 'pino nowar')
# =============================================================================
# Function `randomStep' constructs the dictionary `steps'.
# Each unique token is a key, its values are the tokens immediately following
# each occurance of a key, e.g., the sequence 'the wine is' provides the key 'the'
# with value 'is', and key 'wine' with value 'is'. Keys may contain multiple identical
# values; this reflects the probability of selecting a ertain value given a key.
# This is the anture of a Markov chain: the probability of transitioning from one state
# to another. Here, the probability of transitioning from one word to another stems
# from the relative frequency of that bi-gram.
# =============================================================================

def randomStep(tokens):
    # Build dictionary to represent transition matri - possible steps to take
    steps = {}
    for k in range(len(tokens)-1):
        token = tokens[k]
        if token in steps:
            steps[token].append(tokens[k+1])
        else:
            steps[token] = [tokens[k+1]]
    return steps
# =============================================================================
# Generate wine review by a random walk through our dictionary 
# =============================================================================

def randomWalk(steps):
    # Must begin with a word (capitalised)
    words = [w for w in list(steps) if w.isalpha()]
    token = choice(words) # token requires defining for while loop
    walk = token.capitalize()
    numtokens = 1 # counter for number of tokens in Markov chain
    while True:
        token = choice(steps[token]) # take random step
        # Special characters lose their special meaning inside sets []
        if match('[?.!]',token): # if end of sentence
            # Append token to walk without whitespace
            walk += token
            # If number of tokens in Markov chain exceeds set tolerance of 50,
            # then terminate Markov process.
            if numtokens > 50:
                break
            # Otherwise, begin new sentence
            token = choice(words)
            walk += ' ' + token.capitalize()
            # This is actually a discontinuity in the Markov chain (violating the Markov process)
            # In doing so, each sentence begins with a word, and is then capitalised.
        elif match('[,;:%]',token): # else if sentence break: `token-[,;:%]-whitespace'
            walk += token
        else: # else if: token-whitespace-token
            walk += ' ' + token
        numtokens += 1
    # returns Markov chain generated wine review. May contain punctuation errors
    # as function is vulnerable to certain syntactics, e.g., `(-whitespace-token'
    return walk
# =============================================================================
# Example:
# =============================================================================

minedText = mineText(filteredData, 'france', 'pino nowar')
tokens = word_tokenize(minedText)
steps = randomStep(tokens)
walk = randomWalk(steps)
print(walk)