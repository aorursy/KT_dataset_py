# Import Library 

import nltk
# Now download necessary text files 

# First press d then 'all'

# After completing download press 'q' to exit 



# nltk.download()
# Now import all books what we have downloaded 

from nltk.book import * 
# Now check all books 

text1



# It will only shows the book author and published year 
# What we neeed



# urllib

# urlopen()

# read()



from urllib.request import urlopen

url = "https://www.thedailystar.net/business/news/can-bangladesh-challenge-vietnam-japans-ict-market-1977097"

raw = urlopen(url).read()
# Return xml text 

raw
type(raw)
len(raw)
raw_data = raw.decode('utf-8')
len(raw_data)
type(raw)
tokens = nltk.word_tokenize(data)
# Concord Functions Test 

text2.concordance('love')



# It will shows all context which has word 'love'
## Similer Test 

text2.similar('daughter')

## Shows all similer word to 'daughter'
## common_context text

text2.common_contexts(["father", "mother"])
# Okay now  read a text file called news 

# A text file downloaded into .txt files 

data = open('../input/newsbd/news.txt', encoding='latin2').read()
data
type(data)
# now create tokenization 

tokens = nltk.word_tokenize(data)
type(tokens)
# After applying it every words even space were seperated with a cotetation. 

tokens
# Here some of the words has upper and lower case. For better understanding lets make all upper case word into lower case word 

words = [w.lower() for w in tokens]
# All word are now lower case 

words
# Okay fine, now sorting the text 

# But there could be duplicated words. So, at first remove thoese duplicated words



print('Without removing duplicated words total length = ', len(words))



vocabs = sorted(set(words))

print('After removing duplicated words total length = ', len(vocabs))
import re 
# Corpus is a collection of text 

# Create a word list fro corpus data 

wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]



# Search data in the list end with a

[w for w in wordlist if re.search('a$', w)]
# PorterStammer() : Remove extra words 

# LancasterStammer(): Correct spelling 
row_data = open('../input/newsbd/news.txt', encoding='latin2').read()
row_data
tokens = nltk.word_tokenize(row_data)
# Now Creating stammer 

p = nltk.PorterStemmer() 

l = nltk.LancasterStemmer()
[p.stem(t) for t in tokens]
[l.stem(t) for t in tokens]
# It will not removing any fefx (last extension of the words) and each words has a meaning

le = nltk.WordNetLemmatizer()

[le.lemmatize(t) for t in tokens]
# load segmenter 

sen_segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
# load data from corpus 

text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
# Find sentences 

from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(text)
sentences[171:181]
def segment(text, segs):

    words = []

    last = 0

    

    for i in range(len(segs)):

        if segs[i] == '1':

            words.append(text[last: i+1])

            last = i+1

    words.append(text[last:])

    return words
# Create words and their segmenter 

text = 'HelloThere.IamHappytoseeyou.'

seg1 = '0000000000100000000000000001'

seg2 = '000010000011010000101001001'
# apply functions

segment(text, seg1)
segment(text, seg2)
text = """ The most important source of texts is undoubtedly the Web."""
sents = sent_tokenize(text)
sents
# word tokenization 

sents = [nltk.word_tokenize(sents) for sents in sents]
sents
sents = [nltk.pos_tag(sents) for sents in sents]
sents