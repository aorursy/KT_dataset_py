



import nltk







from nltk.book import*



text1.concordance("india")




text2.concordance("good")







text1.similar("india")







text2.common_contexts(['india','good'])



text1.dispersion_plot(['india','good'])
from nltk.tokenize import word_tokenize, sent_tokenize

text="this is pune. pune is a clean city"

sents=sent_tokenize(text)

print(sents)
l = nltk.word_tokenize(text)

print(l)
from nltk.corpus import stopwords 

from string import punctuation

customStopWords=set(stopwords.words('english')+list(punctuation))

wordsWOStopwords=[word for word in word_tokenize(text) if word not in customStopWords]

print(wordsWOStopwords)
text2="this is BVCOEW" 

from nltk.stem.lancaster import LancasterStemmer

st=LancasterStemmer()

stemmedWords=[st.stem(word) for word in word_tokenize(text2)]

print(stemmedWords)




nltk.pos_tag(word_tokenize(text2))


