from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# Show Python version

import platform

platform.python_version()

import multiprocessing



multiprocessing.cpu_count()
# In[2]:





try:

    import scrapy

except:

    get_ipython().system('pip install scrapy')

    import scrapy

import scrapy.crawler as crawler





# In[3]:





import json

class JsonWriterPipeline(object):



    def open_spider(self, spider):

        self.file = open('quoteresult.txt', 'w')



    def close_spider(self, spider):

        self.file.close()



    def process_item(self, item, spider):

        line = json.dumps(dict(item)) + "\n"

        self.file.write(line)

        return item



import logging

from multiprocessing import Process, Queue

from twisted.internet import reactor
class QuotesSpider(scrapy.Spider):

    # Spider name

    name = 'amazon_reviews'

    # Domain names to scrape

    allowed_domains = ['amazon.in']

    # Base URL for the MacBook air reviews

    myBaseUrls = [

                 "https://www.amazon.in/OnePlus-Display-Storage-4085mAH-Battery/product-reviews/B07DJ8K2KT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber=",

                "https://www.amazon.in/Apple-iPhone-XR-128GB-Black/product-reviews/B07JG7DS1T/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber=",

        "https://www.amazon.in/Samsung-Galaxy-M30s-Blue-Storage/product-reviews/B07HGMQX6N/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=",

        "https://www.amazon.in/Redmi-Note-Pro-Storage-Processor/product-reviews/B07X1KT6LW/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=",

        "https://www.amazon.in/Apple-iPhone-11-64GB-Black/product-reviews/B07XVMDRZY/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=",

        "https://www.amazon.in/Apple-iPhone-7-32GB-Black/product-reviews/B01LZKSVRB/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="

                 ]

    start_urls=[]

    custom_settings = {

        'LOG_LEVEL': logging.WARNING,

        'ITEM_PIPELINES': {'__main__.JsonWriterPipeline': 1}, # Used for pipeline 1

        'FEED_FORMAT':'json',                                 # Used for pipeline 2

        'FEED_URI': 'qa.json'                        # Used for pipeline 2

    }

    

    

    # Creating list of urls to be scraped by appending page number a the end of base url

    for myBaseUrl in myBaseUrls:

        for i in range(1,121):

            start_urls.append(myBaseUrl+str(i))

            

    

    # Defining a Scrapy parser

    def parse(self, response):

            data = response.css('#cm_cr-review_list')

             

            # Collecting product star ratings

            star_rating = data.css('.review-rating')

             

            # Collecting user reviews

            comments = data.css('.review-text')

            count = 0

             

            # Combining the results

            for review in star_rating:

                yield{'stars': ''.join(review.xpath('.//text()').extract()),

                      'comment': ''.join(comments[count].xpath(".//text()").extract())

                     }

                count=count+1





def run_spider(spider):

    def f(q):

        try:

            runner = crawler.CrawlerRunner()

            deferred = runner.crawl(spider)

            deferred.addBoth(lambda _: reactor.stop())

            reactor.run()

            q.put(None)

        except Exception as e:

            q.put(e)



    q = Queue()

    p = Process(target=f, args=(q,))

    p.start()

    result = q.get()

    p.join()



    if result is not None:

        raise result

run_spider(QuotesSpider)
import pandas as pd

dfjson = pd.read_json('qa.json')

dfjson

for i in range(dfjson.shape[0]):

    dfjson.iloc[i][1]=dfjson.iloc[i][1].strip()

dfjson

print(dfjson.isnull().any(axis = 0))
features = dfjson['comment']
features

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

features = features.apply(lambda x: x.split())

features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(max_features = 5000)

features = list(features)

features = tv.fit_transform(features).toarray()

features
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

features=sc.fit_transform(features)
features

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y= le.fit_transform(dfjson['stars'])

Y
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



features_train, features_test, labels_train, labels_test = train_test_split(features, Y, test_size = .05, random_state = 0)
# Using linear support vector classifier

lsvc = LinearSVC()

# training the model

lsvc.fit(features_train, labels_train)

# getting the score of train and test data

print(lsvc.score(features_train, labels_train)) # 90.93

print(lsvc.score(features_test, labels_test))   # 83.75

# model 2:-

# Using Gaussuan Naive Bayes

gnb = GaussianNB()

gnb.fit(features_train, labels_train)

print(gnb.score(features_train, labels_train))  # 78.86

print(gnb.score(features_test, labels_test))    # 73.80
# Logistic Regression

from sklearn.neural_network import MLPClassifier

lr = LogisticRegression()

lr.fit(features_train, labels_train)

print(lr.score(features_train, labels_train))   # 88.16

print(lr.score(features_test, labels_test))     # 83.08

# model 4:-

# Random Forest Classifier

rfc = RandomForestClassifier(n_estimators = 10, random_state = 0)

rfc.fit(features_train, labels_train)

print(rfc.score(features_train, labels_train))  # 98.82

print(rfc.score(features_test, labels_test))    # 79.71
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=3000,activation = 'relu',solver='adam',random_state=1)

classifier.fit(features_train, labels_train)

print(classifier.score(features_train, labels_train))  # 98.82

print(classifier.score(features_test, labels_test)) 
try:

    from gensim.test.utils import common_texts,get_tmpfile

    from gensim.models import Word2Vec

except:

    get_ipython().system('pip install gensim')

    from gensim.test.utils import common_texts,get_tmpfile

    from gensim.models import Word2Vec
strin=[]

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

for i in range(dfjson.shape[0]):

    no_punct = ""

    for char in dfjson.iloc[i][1]:

        if char not in punctuations:

            no_punct = no_punct + char

        else:

            no_punct+=' '

    strin.append(no_punct.lower().split())
model=Word2Vec(strin,size=5000, window=5, min_count=5)

model.wv.most_similar("galaxy")
model.wv.most_similar("7")
model.wv.most_similar("apple")
string="i liked the apple phone"

string=string.split()



res=""

for word in string:

    res+=ps.stem(word)+' '

res

res=[res]
ress=tv.transform(res).toarray()

ress
resss=sc.transform(ress)
resss
lr.predict(resss)