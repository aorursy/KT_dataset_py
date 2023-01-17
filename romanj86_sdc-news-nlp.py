# Install the libraries needed for getting a list of URLs and for extracting text from articles

!pip install newspaper3k

!pip install newsapi-python
# Import the article-extractor package

from newspaper import Article 
# Loop example

numbers = [1,2,3,4,5]

for some_happy_number in numbers:

    print(some_happy_number * 2)
# Create a list [] called "urls" with 2 urls leading to some news articles

urls = ['https://techcrunch.com/2019/04/08/iphone-spyware-certificate/', 

 "https://techcrunch.com/2019/04/07/rise-of-the-snapchat-empire/"]
# Extract the article text

article_container = [] #create an empy list



for happy_url in urls: #take one url at a time

    our_happy_test_article = Article(happy_url) #instantiate it as an "Article"

    our_happy_test_article.download() #download it

    our_happy_test_article.parse() #read it (and try to guess what the title, author etc. are)

    article_container.append(our_happy_test_article.text) #extract its text and put it (append) into the empty list created earlier
from newsapi import NewsApiClient #import news-api

from collections import Counter #import the counter module, which allows to count stuff (useful)

import itertools #iterator library that helps performing complex iteration routines (e.g. combinations)
# for example: give me all possible combinations of 2 elements from 1,2,3



list(itertools.combinations([1,2,3], 2))
# identify with the server...



# GET your free API key at https://newsapi.org/



newsapi = NewsApiClient(api_key='XXXXXXX12345')
# Let's fetch urls for 100 most relevant articles for the query: "China Artificial Intelligence"

# As you can see, you have many other options inlcuding language and dates

all_articles = newsapi.get_everything(q='China Artificial Intelligence',

                                        #domains = "techcrunch.com",

                                        language='en',

                                        sort_by='relevancy',

                                        page_size = 100,

                                        #from_param = start_date,

                                        #to = end_date

                                     )
# This will display the url of the first article that has been found - Python indices start with 0, R starts with 1



all_articles['articles'][0]['url']
# here we collect all urls into one list.

# the below is a list comprehension - a short option in Python to write a loop.

# it can be translated into: *Create a list in which you pyt the url that you strip from

# each element in all_articles['articles']*



urls_big = [x['url'] for x in all_articles['articles']]

# Let's fetch all the 100 articles



texts = []



for url in urls_big:

    article = Article(url)

    article.download()

    try:

      article.parse()

    except Exception as e:

      print(e)

      continue

    texts.append(article)
# texts seems to be a list of objects that are not purely text but als contain other meta-information

# let's make sure that only the text is left

texts = [x.text for x in texts]
# quick check of how long they are

len(texts)
# downlaod the medium size-model if you work on your computer or google colab (or elsewhere) for now we comment that out 

# because Kaggle has us covered with the large model

#!python -m spacy download en_core_web_md
# Introducing spacy



import spacy #load the library

nlp = spacy.load('en_core_web_lg') #load the (larg english) model
# Let's try out some stuff



# product 3 sentences

sen1 = "The weather today is cold and Donald Trump is fun."

sen2 = "It's sunny and im HAPPY"

sen3 = "Everyone is bored and cold"
# Let spacy read and annotate them



AI_sen1 = nlp(sen1)

AI_sen2 = nlp(sen2)

AI_sen3 = nlp(sen3)
# Getting the 2nd entity type of the first sentence

AI_sen1.ents[1].label_
# let's have it read one of our articles



AI_texts_0 = nlp(texts[0])
#Make a list of entity-texts from all entities in text 0 if the entity is a person



[ent.text for ent in AI_texts_0.ents if ent.label_ == 'PERSON']
# lets extract all (location, person, orga : GPE, PERSON, ORG) entities into an empty container



container = []



for article in texts: # take an article

    article_nlp = nlp(article) #read it

    entities = [ent.text for ent in article_nlp.ents if ent.label_ == 'GPE'] # extract entities for the single articles

    container.extend(entities) # drop them into the "container"
people = Counter(container) #count up stuff in the container

people.most_common(100) #show most common 100
org = Counter(container)

org.most_common(100)
gpe = Counter(container)

gpe.most_common(100)