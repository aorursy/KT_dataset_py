import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
## Importing Libraries
import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
%matplotlib inline
dictOfFilenames={i : filenames[i] for i in range(0, len(filenames) )}

dict_files=dictOfFilenames.copy()
dict_files
for i,filename in enumerate(filenames):
    dictOfFilenames[i] = open(os.path.join(dirname, filename),'r').read()
dictOfFilenames[0]
import nltk
from nltk.corpus import words as english_words, stopwords
import re

## replacing the newlines and extra spaces
corpus = dictOfFilenames[0].replace('\n', ' ').replace('\r', '').replace('  ',' ').lower()

## removing everything except alphabets
corpus_sans_symbols = re.sub('[^a-zA-Z \n]', '', corpus)

## removing stopwords
stop_words = set(w.lower() for w in stopwords.words())

corpus_sans_symbols_stopwords = ' '.join(filter(lambda x: x.lower() not in stop_words, corpus_sans_symbols.split()))
print (corpus_sans_symbols_stopwords)
from nltk.stem import PorterStemmer
stemmer=nltk.PorterStemmer()
corpus_stemmed = ' ' .join (map(lambda str: stemmer.stem(str), corpus_sans_symbols_stopwords.split()))
print (corpus_stemmed)
# Plot top 20 frequent words
from collections import Counter
word_freq = Counter(corpus_stemmed.split(" "))
import seaborn as sns
sns.set_style("whitegrid")
common_words = [word[0] for word in word_freq.most_common(20)]
common_counts = [word[1] for word in word_freq.most_common(20)]


plt.figure(figsize=(12, 8))

sns_bar = sns.barplot(x=common_words, y=common_counts)
sns_bar.set_xticklabels(common_words, rotation=45)
plt.title('Most Common Words in the document')
plt.show()
import spacy
## Spacy example 
nlp = spacy.load('en')
doc = nlp(dictOfFilenames[0])
for token in doc:
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(
        token.text,
        token.idx,
        token.lemma_,
        token.is_punct,
        token.pos_,
        token.tag_
    ))
## passing our text into spacy
doc = nlp(dictOfFilenames[0])

## filtering stopwords, punctuations, checking for alphabets and capturing the lemmatized text
spacy_tokens = [token.lemma_ for token in doc if token.is_stop != True \
                and token.is_punct != True and token.is_alpha ==True]
word_freq_spacy = Counter(spacy_tokens)

# Plot top 20 frequent words

sns.set_style("whitegrid")
common_words = [word[0] for word in word_freq_spacy.most_common(20)]
common_counts = [word[1] for word in word_freq_spacy.most_common(20)]


plt.figure(figsize=(12, 8))

sns_bar = sns.barplot(x=common_words, y=common_counts)
sns_bar.set_xticklabels(common_words, rotation=45)
plt.title('Most Common Words in the document')
plt.show()
text_str = ''.join(dictOfFilenames[0].replace('\n',' ').replace('\t',' '))
sentences_split = text_str.split(".")
sentences_split[67]
doc = nlp(text_str)
sentence_list = [s for s in doc.sents]
sentence_list[67]
spacy.displacy.render(sentence_list[67], style='dep',jupyter=True,options = {'compact':60})
pos_list = [(token, token.pos_) for token in sentence_list[67]]
text_ent_example=dictOfFilenames[0]
doc = nlp(text_ent_example)
spacy.displacy.render(doc, style='ent',jupyter=True)
