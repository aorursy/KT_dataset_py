!pip install scrapy #installing scrapy
import numpy as np

import pandas as pd

import re # for cleaning data

import scrapy 

from scrapy.crawler import CrawlerProcess

from scrapy import Selector

import requests
#This is a normal scraping method without building a crawler class 

'''url="https://pitchfork.com/artists/1339-eminem"

html=requests.get(url).content

sel=Selector(text=html)

album_reviews={}

links_to_the_reviews=sel.xpath('//div[@class="review"]/a/@href').extract()

for link in links_to_the_reviews:

  album_page=requests.get("https://pitchfork.com/"+link).content

  sel_album=Selector(text=album_page)

  review=sel_album.xpath("//div[@class='contents dropcap']//text()").extract()

  name=sel_album.xpath('//h1[@class="single-album-tombstone__review-title"]/text()').extract()[0]

  #print(name)

  for i in range(review.count('\n')):

    review.remove('\n')

  review="".join(review)



  album_reviews[name]=review'''  
class scraper(scrapy.Spider):

  name="scraper"

  def start_requests(self):

    url="https://pitchfork.com/artists/1339-eminem"

    yield scrapy.Request(url=url,callback=self.parse)

  def parse(self,response):

    links=response.css('div.review>a::attr(href)').extract()

    for link in links:

      yield response.follow(url=link,callback=self.parse2)

  def parse2(self,response):

    name=response.css('h1.single-album-tombstone__review-title::text').extract()[0]

    names.append(name)

    review=response.css('div.contents.dropcap ::text').extract()

    reviews.append(review)     
#Starting the crawler to crawl through web pages and extract the data of our need.

reviews=[]

names=[]

process = CrawlerProcess()

process.crawl(scraper)

process.start()
#let us check how our review looks like

reviews[0]
#Joining the data

for i in range(len(reviews)):

  reviews[i]="".join(reviews[i])

  reviews[i]=re.sub("\n"," ",reviews[i])
data=pd.DataFrame(list(zip(names,reviews)),columns=["Album_name","Review"])

data.head()
data["Album_name"] #Albums 
#Importing required libraries

from collections import Counter

import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS 

import spacy

import seaborn as sns

from PIL import Image

nlp=spacy.load('en_core_web_sm')

import logging
Album_name=data.iloc[0,0] #Music to be murdered by

Album_review=data.iloc[0,1] #review of the album
Common_words=[] #This will contain the most frequent words used by reviewers

doc=nlp(Album_review) #Tokenizing the review

tokens=[token.lower_ for token in doc if not token.is_punct and not token.is_stop and token.lower_  not in "eminem"] #Removing stop words 

#and punctuations

count=Counter(tokens) 

count=count.most_common(40)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

labels=[tup[0] for tup in count]

value=[tup[1] for tup in count]

plt.figure(figsize=(10,5))

plot=sns.barplot(x=labels,y=value)

plot.set_xticklabels(labels=labels,rotation=90)

plot.set_title("Review:Music To Be Murdered By(most frequent words)")

#from PIL import Image

#from google.colab import files

#image=files.upload() # I made this notebook on colab , you might be on a different platform so look out when doing this cell.
my_mask=np.array(Image.open("../input/6930355.png"))

cloud=WordCloud(background_color="white",mask=my_mask,stopwords=STOPWORDS)

cloud.generate(data.iloc[0,-1])

#cloud.to_file("Eminem.png")

plt.figure(figsize=(10,10))

plt.imshow(cloud,interpolation='bilinear')

plt.tight_layout()

plt.axis("off")