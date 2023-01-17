#importing all neccesary libraries
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.book import *
texts = pd.read_csv("../input/train.csv")
# getting lenght of the text in words and punctuation symbols
print('Words + punctuation:', len(text4))

# getting all the distinc words in text
set(text4)

# we can also combine them and see how many distinct words are exactly there 
distinct_words = len(set(text4))
print('Distinc words and punctuation symbols:', distinct_words)

# based on that we can determine how 'Rich' is the text
rich = len(set(text4)) / len(text4)
print('My book is rich ratio is', rich)

# what part of the text takes specific word in percents?
text4.count("I")
word_percent = 100 * text4.count('I') / len(text4)
print('Authors talks about himselft', word_percent, 'percent of time!' )

# This will help to get the most repetetive words and their count
fdist = FreqDist(text4)
print(fdist.most_common(5))

### HELPFUL FUNCTIONS TO DO THE SAME FASTER
def lexical_diversity(text):
    return len(set(text)) / len(text) [2]

def percentage(count, total):
    return 100 * count / total
# Now while those words do not make much sense, we should determine which ones are just common words for every text
plt.figure(figsize=(15 , 8))
fdist.plot(50, cumulative=True)

# This plot basically shows word position in the text. We can see here that the word "please" was used only once at the end of the text.
plt.figure(figsize=(14 , 8))
text4.dispersion_plot(["love", "hate", "duty", "tax", "please"])
# getting all words that do not repeat itself
hapaxes = fdist.hapaxes()
print('Unique words:')
print(hapaxes[:8])
# getting combined
combined = sorted(w for w in set(text4) if len(w) > 7 and fdist[w] > 7)
print('--------------------------------------------')
print('Counting words that are longer than 7 symbols and are repeated at least 7 times:')
print(combined[:10])
#lets get the collocation
print('--------------------------------------------')
print('Two consecutive words (bigrams) that get repeated most of the times:')
text4.collocations()

# Thats how we can find any particular word
concordance = text4.concordance("please")
print(concordance)
print('--------------------------------------------')
text1.similar("love")
print('and')
text2.similar('love')
print('--------------------------------------------')
# Now. Thats the list of words which were used in there same context as 'think'! That is actually very interesting.
text1.common_contexts(["love", "sea"])
from nltk.corpus import gutenberg
gutenberg.fileids()
#picking up texts from the Projest Gutenberg electronic Text archive
nltk.corpus.gutenberg.fileids()
# After we get a list of names we can put in argument just the text we need. Notice .words
hamlet = gutenberg.words('shakespeare-hamlet.txt')
for fileid in gutenberg.fileids():
    # contents of the file without any linguistic processing
     num_chars = len(gutenberg.raw(fileid)) 
    # average word lenght (!it counts spaces, so you have to assume number is 1 less)
     num_words = len(gutenberg.words(fileid))
    # average sentence length
     num_sents = len(gutenberg.sents(fileid))
     num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
     print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)
from nltk.corpus import inaugural
# extracting the first four characters, using fileid[:4] to get the year 
[fileid[:4] for fileid in inaugural.fileids()]
cfd = nltk.ConditionalFreqDist(
           (target, fileid[:4])
           for fileid in inaugural.fileids()
           for w in inaugural.words(fileid)
           for target in ['america', 'me']
           if w.lower().startswith(target)) 
cfd.plot()
from nltk.corpus import PlaintextCorpusReader
corpus_root = '../input/'
wordlists = PlaintextCorpusReader(corpus_root, '.*') 
wordlists.words('test.csv')
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
           (genre, word)
           for genre in brown.categories()
           for word in brown.words(categories=genre))

