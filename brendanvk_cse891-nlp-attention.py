# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
with open('/kaggle/input/europarl/europarl_raw/ep-00-02-02.en','r') as f:

    for i, line in enumerate(f):

        print(line)

        if i > 1:

            break

with open('/kaggle/input/europarl/europarl_raw/ep-00-02-02.es','r') as f:

    for i, line in enumerate(f):

        print(line)

        if i > 1:

            break
def piglatin(tokens):

    tokens_clean = map(lambda token: token.lower().strip(), tokens)

    

    def is_vowel(char):

        return char in {'a', 'e', 'i', 'o', 'u'}

    

    def piglatin_word(word):

        if is_vowel(word[0]):

            # Vowel

            return word + 'way'

        elif word[0].isalpha():

            # Consonant

            return word[1:] + word[0] + 'ay'

        else:

            # Unknown, punctuation probably

            return word

    

    return map(piglatin_word, tokens_clean)
' '.join(piglatin("This will make any iterable collection of tokens into pig latin .".split()))