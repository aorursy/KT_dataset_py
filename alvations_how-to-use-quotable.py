from string import punctuation

from collections import Counter



import pandas as pd



from nltk import word_tokenize

from nltk.corpus import stopwords



# This is a short list of common English contractions.

contractions = "n't 've 'm 're 'd 'll 's".split()

# Stopwords are non-content words that has little/no semantics.

stoplist = stopwords.words() + list(punctuation) + contractions
quotables = pd.read_csv('../input/author-quote.txt', delimiter='\t', header=None)

quotables = quotables.rename(columns={0:'author', 1:'quote'})
# Get the list of authors that are available in Quotable.

authors = sorted(set(quotables['author']))
obama_quotes = quotables[quotables['author'] ==  'Barack Obama']

obama_quotes.head()
# Using the DataFrame.apply() function, it's easy:

obama_quotes_lowered = obama_quotes['quote'].apply(str.lower)
def remove_stopwords(text):

    """ Function to remove stop words. """

    for word in word_tokenize(text):

        if word not in stoplist:

            yield word



# Lower case the quotes.

obama_quotes_lowered = obama_quotes['quote'].apply(str.lower)



# Apply the `remove_stopwords()` function.

# Since the function returns a generator (from "yield"), 

# we can simply cast the column of each row column with

# `DataFrame.apply(list)`

obama_quotes_lowered_nostop = obama_quotes['quote'].apply(str.lower).apply(remove_stopwords).apply(list)



obama_quotes_lowered_nostop.head()