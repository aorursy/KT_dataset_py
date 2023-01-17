'''
AUTO SUMMARIZE TEXT

Strategy

1. extract information from web
2. clean raw data (html data)
3. sentense tokenize
4. word tokenize
5. remove stop words
6. find the most important words (important words are generally more often repeated)
7. calculate a score for sentences containing important words
8. show top waited sentances in order


'''
'''
Importing all required libraries
'''
from urllib.request import urlopen
from bs4 import BeautifulSoup


import nltk #natural language tool kit, it has lot useful functions to do NLP
from nltk.tokenize import sent_tokenize, word_tokenize # to break big paragraph to sentance and words
from nltk.corpus import stopwords # list of stopwords
from nltk.probability import FreqDist # find word repeat cout
from collections import defaultdict # dictionary
from string import punctuation # extract punctuation to remove from text. we remove stopwords + punctuations
from heapq import nlargest #
'''
Get data
Here I taken news article from IHiS website
'''

articleURL = "https://www.ihis.com.sg/Latest_News/News_Article/Pages/Chatbots-show-the-way-in-using-tech-to-boost-healthcare-.aspx"

# read article using urlopen, Full html will be store to page object
page = urlopen(articleURL).read().decode('utf8','ignore')
#print(page)
# using BeautifulSoup we convet to lxml format. advantage of this is we can read each individual markup using tags

soup = BeautifulSoup(page,'lxml')
#print(soup)
# get artile title and print
article_title = soup.find("article").h1.text
print("Title of the article::",article_title)
print("")

# get article body using class and div
article_body = soup.find("div",class_="ms-rtestate-field")

# when I saw html markup, first 4 line are img and img title. So skipping those from the body
text = ' '.join(map(lambda p: p.text, article_body.find_all('p')[4:-9]))

print(text)
# encoding text to ascii format and removing ? from the text

text = text.encode('ascii',errors='replace').decode().replace('?','')

text
# Using sent_tokenize deviding entire text to sentences

sents = sent_tokenize(text)
sents
print(len(sents))
# word tokenizing using word_tokenize

words = word_tokenize(text.lower())
words
# Preparing stopwords list to remove from the text

stopwords = set((stopwords.words('english')+list(punctuation)))
stopwords
# Removing stop words from words list

words = [word for word in words if word not in stopwords]
words
# caliculating word frequecy using FreDist function

freq = FreqDist(words)
freq
#type(freq)
freq['healthcare']
# Caliculating each sentence wait using words in the sentence.

ranking = defaultdict(int)
    
for i, sent in enumerate(sents):
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]
#geting top 5 waited values

sentsIDX = nlargest(5,ranking,key=ranking.get)
sentsIDX
for i in sentsIDX:
    print(ranking.get(i))
# Order the list and printing
article_title = soup.find("article").h1.text
print("Title:",article_title)
print("")


for j in sorted(sentsIDX):
    print(sents[j])

