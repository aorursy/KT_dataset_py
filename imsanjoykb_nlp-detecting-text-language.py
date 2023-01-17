text = "Yo man, it's time for you to shut yo' mouth! I ain't even messin' dawg."
import sys



try:

    from nltk.tokenize import wordpunct_tokenize # RE-based tokenizer which splits text on whitespace and punctuation (except for underscore)

except ImportError:

    print('[!] You need to install nltk (http://nltk.org/index.html)')
test_tokens = wordpunct_tokenize(text)

test_tokens
from nltk.corpus import stopwords

stopwords.readme().replace('\n', ' ') # Since this is raw text, we need to replace \n's with spaces for it to be readable.
stopwords.fileids() # Most corpora consist of a set of files, each containing a piece of text. A list of identifiers for these files is accessed via fileids().
stopwords.raw('greek')
stopwords.raw('greek').replace('\n', ' ') # Better
stopwords.words('english')[:10]
stopwords.sents('greek')
len(stopwords.words(['english', 'greek'])) # There is a total of 444 Greek and English stop words
language_ratios = {}



test_words = [word.lower() for word in test_tokens] # lowercase all tokens

test_words_set = set(test_words)



for language in stopwords.fileids():

    stopwords_set = set(stopwords.words(language)) # For some languages eg. Russian, it would be a wise idea to tokenize the stop words by punctuation too.

    common_elements = test_words_set.intersection(stopwords_set)

    language_ratios[language] = len(common_elements) # language "score"

    

language_ratios
most_rated_language = max(language_ratios, key=language_ratios.get) # The key parameter to the max() function is a function that computes a key. In our case, we already have a key so we set key to languages_ratios.get which actually returns the key.

most_rated_language
test_words_set.intersection(set(stopwords.words(most_rated_language))) # We can see which English stop words were found.