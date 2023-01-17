import pandas as pd

import numpy as np



s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234','SteveSmith'])

s
s.str.lower()
s.str.upper()
s = pd.Series(['Tom ', ' William Rick', 'John', 'Alber@t'])

print (s)

print ("After Stripping:")

print (s.str.strip())
s.str.cat(sep='_')
s.str.isnumeric()




time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 

                  "Tuesday: The dentist's appointment is at 11:30 am.",

                  "Wednesday: At 7:00pm, there is a basketball game!",

                  "Thursday: Be back home by 11:15 pm at the latest.",

                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]



df = pd.DataFrame(time_sentences, columns=['text'])

df
# find the number of characters for each string in df['text']

df['text'].str.findall()
# find the number of tokens for each string in df['text']

df['text'].str.split().str.len()
# find which entries contain the word 'appointment'

df['text'].str.contains('appointment')
# find how many times a digit occurs in each string

df['text'].str.count(r'\d')
df['text'].str.findall(r'(\d?\d):(\d\d)')
# replace weekdays with '???'



df['text']
# extract the entire time, the hours, the minutes, and the period



df['text'].str.extractall(r'(?P<time>\d:\d{1,2})')
# extract the entire time, the hours, the minutes, and the period with group names

df['text'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')
import nltk



nltk.download('book')

from nltk.book import *
text7
sent7
# how many word in text 7 ? 

len(text7)
# how many unique word in text 7 

len(set(text7))
list(set(text7))[:10]
from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
text = '  '.join([j for j in text7 ])

wordcloud = WordCloud().generate(text)

# Display the generated image:

plt.figure(figsize = (15, 15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# lower max_font_size, change the maximum number of word and lighten the background:

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)

plt.figure()

plt.figure(figsize = (15, 15))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# Create stopword list:

stopwords = set(STOPWORDS)

# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

plt.figure(figsize = (15, 15))

# Display the generated image:

# the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
dist = FreqDist(text7)

len(dist)
vocab1 = dist.keys()

#vocab1[:10] 

# In Python 3 dict.keys() returns an iterable view instead of a list

list(vocab1)[:10]
dist['Vinken']
freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]

freqwords
input1 = "List listed lists listing listings"

words1 = input1.lower().split(' ')

words1
porter = nltk.PorterStemmer()

[porter.stem(t) for t in words1]
udhr = nltk.corpus.udhr.words('English-Latin1')

udhr[:20]
[porter.stem(t) for t in udhr[:20]] # Still Lemmatization
WNlemma = nltk.WordNetLemmatizer()

[WNlemma.lemmatize(t) for t in udhr[:20]]
text11 = "Children shouldn't drink a sugary drink before bed."

text11.split(' ')
nltk.word_tokenize(text11)
text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"

sentences = nltk.sent_tokenize(text12)

len(sentences)
sentences
nltk.help.upenn_tagset('MD')
text13 = nltk.word_tokenize(text11)

nltk.pos_tag(text13)
text14 = nltk.word_tokenize("Visiting aunts can be a nuisance")

nltk.pos_tag(text14)
nltk.help.upenn_tagset('VBG')
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