# Checking System Version

import sys

print(sys.version)
from nltk.book import *
# Recall text by typing their name only

text7
# It gives you list as per the nltk book import

texts()
# Searching text

text1.concordance("monstrous") 
text2.concordance("affection")
text1.similar("monstrous") 
text2.similar("monstrous")
text2.common_contexts(["monstrous","very"])
text4.dispersion_plot(["citizens","democracy","freedom","duties","America"])
print(len("nishant;., is here"))



List_N1 = ['Call', 'me', 'Nishant', '.']

print(len(List_N1))



print(len(text3))
set(text3)
len(set(text3))
sorted(set(text3))
len(text3)/len(set(text3))  #tokens (individual words and punctuation) occur in a given text, divided by how many types (unique words and punctuation)
print(text5.count('lol')) # count of a specific word.

100*(text5.count('lol'))/len(text5) # percentage against total word count.
#tokens (individual words and punctuation) occur in a given text, divided by how many types (unique words and punctuation)

def lex_dive(text):

    return len(text)/len(set(text))
def percentage(count, total):

    return 100*count/total
lex_dive(text3)
percentage(text5.count('lol'),len(text5))
# We can inspect the total number of words (“outcomes”) that have been counted up using FreqDist

fdist1 = FreqDist(text1)

fdist1
#The expression keys() gives us a list of all the distinct types in the text

vocab1 = fdist1.keys()

vocab1
# Count of word 'whale' in the frequency distribution list

fdist1['whale']
# To set figure size of the plot

from matplotlib.pyplot import figure

figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')



# Cumulative Frequency Plot for 50 words

fdist1.plot(50,cumulative = True)
fdist1.hapaxes()
V = set(text1)

long_words = [w for w in V if len(w) > 15]

sorted(long_words)
fdist5 = FreqDist(text5)

sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7)
from nltk import bigrams

list(bigrams(['more', 'is', 'said', 'than', 'done']))
bigrams(['more', 'is', 'said', 'than', 'done'])
text4.collocations()
[len(w) for w in text1]  #  A list of the lengths of words in text1
fdist = FreqDist(len(w) for w in text1) # Frequency distribution of lengths of words in text1

print(fdist)

fdist
print(fdist.most_common()) # A list of frequency disctribution of lengths of words



print(fdist.max()) # Most frequent word length



print(fdist[3]) # Frequncy of word length 3



fdist.freq(3) # Percent Frequency of word length 3
sorted(w for w in set(text1) if w.endswith('ableness')) # Operator endswith
sorted(term for term in set(text4) if 'gnt' in term) # words with substring 'gnt'
sorted(item for item in set(text6) if item.istitle()) # titlecased words
sorted(item for item in set(sent7) if item.isdigit()) #test if s is non-empty and all characters in s are digits
sorted(w for w in set(text7) if '-' in w and 'index' in w)  # Words with two substring(s)/character(s)
sorted(wd for wd in set(text3) if wd.istitle() and len(wd) > 10)  # Titlecased words with more than 10 characters
sorted(w for w in set(sent7) if not w.islower()) # test if s contains cased characters and all are lowercase. Here with not
[len(w) for w in text1]

[w.upper() for w in text1]  # test if s contains cased characters and all are uppercase
sent1 = ['Call', 'me', 'Nishant', '.']



# Looping with Condition using 'for' & 'if'

for xyzzy in sent1:

    if xyzzy.endswith('t'):

        print(xyzzy)