# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import nltk

from nltk.book import*
text7
len(text7)
sents()
len(set(text7))
list(set(text7))[:10]
freq = FreqDist(text7)

freq
freq[',']
key = freq.keys()

list(key)[:10]
freqWords = [words for words in key if len(words)>5 and freq[words]>100]

freqWords
input1 = 'Go go Going Goings Goes'

word1 = input1.lower().split(' ')

word1
porter = nltk.PorterStemmer()

[porter.stem(i) for i in word1]
corpus = nltk.corpus.udhr.words('English-Latin1')

corpus
# still lematization

[porter.stem(t) for t in corpus][:20]
WNlemma = nltk.WordNetLemmatizer()

[WNlemma.lemmatize(t) for t in corpus[:20]]
## Tokenization

text = 'hey whats going on.'

text.split(' ')
nltk.word_tokenize(text)
text12 = "This is the first sentence. A gallon of milk in the Nepal costs Rs.300. Is this the third sentence? Yes, it is!"

sentences = nltk.sent_tokenize(text12)

sentences
len(sentences)
nltk.help.upenn_tagset('MD')
text13 = nltk.word_tokenize(text)

nltk.pos_tag(text13)
text14 = nltk.word_tokenize("Chilling with friends is a fantastic feeling.")

nltk.pos_tag(text14)
# Parsing sentence structure

text15 = nltk.word_tokenize("Alice loves Bob")

grammar = nltk.CFG.fromstring("""

S -> NP VP

VP -> V NP

NP -> 'Alice' | 'Bob'

V -> 'loves'

""")



parser = nltk.ChartParser(grammar)

trees = parser.parse_all(text15)

for tree in trees:

    print(tree)
text18 = nltk.word_tokenize("The old man the boat")

nltk.pos_tag(text18)
text19 = nltk.word_tokenize("Colorless green ideas sleep furiously")

nltk.pos_tag(text19)