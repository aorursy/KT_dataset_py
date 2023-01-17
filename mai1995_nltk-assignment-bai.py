import nltk

#natural language toolkit
from nltk.book import*

#9 types of text available in NLTK book you may choose any text as per your interest
# display all occurences of word in piece of text along with context

text1.concordance("india")
text2.concordance("good")
# returns a list of words that appaer in similar context usually synonyms

text1.similar("india")



text2.similar("good")
# returns contexts shared by 2 words

text2.common_contexts(['india','good'])

# print plot of all occurences of the word relative to begining of the text

text1.dispersion_plot(['india','good'])
from nltk.tokenize import word_tokenize, sent_tokenize

text="welcome to BVCOEW,Pune"

sents=sent_tokenize(text)

print(sents)

l= nltk.word_tokenize(text)

print(l)
# Let's filter out stopwords (words that are very common like 'was', 'a', 'as etc)

from nltk.corpus import stopwords 

from string import punctuation

customStopWords=set(stopwords.words('english')+list(punctuation))

#Notice how we made the stopwords a set



wordsWOStopwords=[word for word in word_tokenize(text) if word not in customStopWords]

print(wordsWOStopwords)
text2="Life is a short journey."

# 'close' appears in different morphological forms here, stemming will reduce all forms of the word 'close' to its root

# NLTK has multiple stemmers based on different rules/algorithms. Stemming is also known as lemmatization. 

from nltk.stem.lancaster import LancasterStemmer

st=LancasterStemmer()

stemmedWords=[st.stem(word) for word in word_tokenize(text2)]

print(stemmedWords)
#NLTK has functionality to automatically tag words as nouns, verbs, conjunctions etc

nltk.pos_tag(word_tokenize(text2))