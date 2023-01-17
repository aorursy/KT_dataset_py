from IPython.display import YouTubeVideo



YouTubeVideo('HctunZLmc10', width=800/1.2, height=450/1.25)
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import re

import os

import string

import pickle





import nltk

import gensim



from nltk.tokenize import word_tokenize, sent_tokenize

from gensim.parsing.preprocessing import remove_stopwords

from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer



from gensim.test.utils import common_corpus, common_dictionary

from gensim.similarities import MatrixSimilarity



from gensim.test.utils import datapath, get_tmpfile

from gensim.similarities import Similarity



from IPython.display import display, Markdown, Math, Latex, HTML





import pandas as pd

import seaborn as sns



!pip install webdriverdownloader

from webdriverdownloader import GeckoDriverDownloader



!pip install selenium

from selenium.webdriver.common.by  import By as selenium_By

from selenium.webdriver.support.ui import Select as selenium_Select

from selenium.webdriver.support.ui import WebDriverWait as selenium_WebDriverWait

from selenium.webdriver.support    import expected_conditions as selenium_ec

from IPython.display import Image







from selenium import webdriver as selenium_webdriver

from selenium.webdriver.firefox.options import Options as selenium_options

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities as selenium_DesiredCapabilities



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()





# YOU MUST ADD YOUR USERNAME AND PASSWORD OF RESEARCH GATE TO THE SECRET CREDENTIALS TO BE ABLE TO GET THE SCRAPED DATA

# email = user_secrets.get_secret("email")

# password = user_secrets.get_secret("pass")

email = user_secrets.get_secret("email")

password = user_secrets.get_secret("pass")

prox_s = user_secrets.get_secret("proxy_server")
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

print("Cols names: {}".format(meta.columns))

meta.head(7)
plt.figure(figsize=(20,10))

meta.isna().sum().plot(kind='bar', stacked=True)
meta.columns
meta_dropped = meta.drop(['who_covidence_id'], axis = 1)
plt.figure(figsize=(20,10))



meta_dropped.isna().sum().plot(kind='bar', stacked=True)
miss = meta['abstract'].isna().sum()

print("The number of papers without abstracts is {:0.0f} which represents {:.2f}% of the total number of papers".format(miss, 100* (miss/meta.shape[0])))
abstracts_papers = meta[meta['abstract'].notna()]

print("The total number of papers is {:0.0f}".format(abstracts_papers.shape[0]))

missing_doi = abstracts_papers['doi'].isna().sum()

print("The number of papers without doi is {:0.0f}".format(missing_doi))

missing_url = abstracts_papers['url'].isna().sum()

print("The number of papers without url is {:0.0f}".format(missing_url))
abstracts_papers = abstracts_papers[abstracts_papers['publish_time'].notna()]

abstracts_papers['year'] = pd.DatetimeIndex(abstracts_papers['publish_time']).year
missing_url_data = abstracts_papers[abstracts_papers["url"].notna()]

print("The total number of papers with abstracts, urls, but missing doi = {:.0f}".format( missing_url_data.doi.isna().sum()))
abstracts_papers = abstracts_papers[abstracts_papers["url"].notna()]
!cat /etc/os-release



!mkdir "/kaggle/working/firefox"

!ls -l "/kaggle/working"

!cp -a "/kaggle/input/firefox-63.0.3.tar.bz2/firefox-63.0.3/firefox/." "/kaggle/working/firefox"

!ls -l "/kaggle/working/firefox"





!chmod -R 777 "/kaggle/working/firefox"

!ls -l "/kaggle/working/firefox"



gdd = GeckoDriverDownloader()

gdd.download_and_install("v0.23.0")

!apt-get install -y libgtk-3-0 libdbus-glib-1-2 xvfb

!export DISPLAY=:99
from selenium.webdriver.common.proxy import Proxy, ProxyType
browser_options = selenium_options()

browser_options.add_argument("--headless")

browser_options.add_argument("--window-size=1920,1080")



proxy = prox_s



capabilities_argument = selenium_DesiredCapabilities().FIREFOX

capabilities_argument["marionette"] = True

capabilities_argument['proxy'] = {

    "proxyType": "MANUAL",

    "httpProxy": proxy,

    "ftpProxy": proxy,

    "sslProxy": proxy

}



browser = selenium_webdriver.Firefox(

    options=browser_options,

    firefox_binary="/kaggle/working/firefox/firefox",

    capabilities=capabilities_argument

)
browser.get("https://www.researchgate.net/login")

browser.find_element_by_id('input-login').send_keys(str(email))

browser.find_element_by_id('input-password').send_keys(str(password))

browser.find_element_by_class_name('action-submit').click()
# ids = browser.find_elements_by_xpath('//*[@class]')
# for ii in ids:

# #     print(ii.tag_name)

#     print (ii.get_attribute('id'))    # id name as string
# browser.find_element_by_class_name('challenge-form').click()
# print(browser.current_url)
# We will be usingn researchgate to scrape more data as follows

# browser.get("https://www.researchgate.net/login")

browser.save_screenshot("screenshot.png")

Image("screenshot.png", width=800, height=500)
def get_citations(top10):

    

    citations = []

    for paper in range(10):

        if(not (type(top10.doi[paper]) == float)):

            browser.get("https://www.researchgate.net/search.Search.html?type=researcher&query=" + str(top10.doi[paper]))



            element = browser.find_elements_by_class_name('nova-c-nav__item-label')

            if(len(element) >= 3):

                numbers = re.findall("\d+" , (element[3].text).lower())

                if len(numbers) >= 1:

                    citations.append(max(int(numbers[0]), 0.1))

                else:

                    citations.append(0.1)

            else:

                citations.append(0.1)   

        else:

            citations.append(0.1)

            

    return citations
porter = PorterStemmer()

lancaster=LancasterStemmer()



# abstracts_only = abstracts_papers['abstract']

# tokenized_abs = []



# for abst in abstracts_only:

#     tokens_without_stop_words = remove_stopwords(abst)

#     tokens_cleaned = sent_tokenize(tokens_without_stop_words)

#     words = [porter.stem(w.lower()) for text in tokens_cleaned for w in word_tokenize(text) if (w.translate(str.maketrans('', '', string.punctuation))).isalnum()]

#     tokenized_abs.append(words)

    

    

# dictionary = gensim.corpora.Dictionary(tokenized_abs)

# corpus = [dictionary.doc2bow(abstract) for abstract in tokenized_abs]

# tf_idf = gensim.models.TfidfModel(corpus)



# tf_idf.save("tfidf")

# dictionary.save("dict")



# with open("corpus.txt", "wb") as fp:

#     pickle.dump(corpus, fp)

with open("/kaggle/input/tfidfcovid19/corpus.txt", "rb") as fp:

    corpus = pickle.load(fp)

    

dictionary = gensim.corpora.Dictionary.load("/kaggle/input/tfidfcovid19/dict")

tf_idf = gensim.models.TfidfModel.load("/kaggle/input/tfidfcovid19/tfidf")
def query_tfidf(query):

    

    query_without_stop_words = remove_stopwords(query)

    tokens = sent_tokenize(query_without_stop_words)



    query_doc = [porter.stem(w.lower()) for text in tokens for w in word_tokenize(text) if (w.translate(str.maketrans('', '', string.punctuation))).isalnum()]



    # mapping from words into the integer ids

    query_doc_bow = dictionary.doc2bow(query_doc)

    query_doc_tf_idf = tf_idf[query_doc_bow]

    

    return query_doc_tf_idf
def rankings(query):



    query_doc_tf_idf = query_tfidf(query)

    index_temp = get_tmpfile("index")

    index = Similarity(index_temp, tf_idf[corpus], num_features=len(dictionary))

    similarities = index[query_doc_tf_idf]



    # Storing similarity in the dataframe and sort from high to low simmilatiry

    print(similarities)

    abstracts_papers["similarity"] = similarities

    abstracts_papers_sorted = abstracts_papers.sort_values(by ='similarity' , ascending=False)

    abstracts_papers_sorted.reset_index(inplace = True)

    

    top20 = abstracts_papers_sorted.head(10)

    top20["doi"].astype(str)

    citations = get_citations(top20)

    top20["citations"] = citations

    top20["similarity"] = top20["similarity"] * (top20.citations / top20.citations.max()) * 0.5

    norm_range = top20['year'].max() - top20['year'].min()

    top20["similarity"] -= (abs(top20['year'] - top20['year'].max()) / norm_range)*0.1

    top20 = top20.sort_values(by ='similarity' , ascending=False)

    top20.reset_index(inplace = True)

    

    return top20
top = rankings("Non-pharmaceutical interventions")
pd.set_option('display.max_colwidth', -1)

top[['index','abstract']].style.set_properties(**{'text-align': "justify"})
pd.set_option('display.max_colwidth', -1)

top[['index','url']].style.set_properties(**{'text-align': "left"})
top = rankings("MERS respiratory sendrom")
pd.set_option('display.max_colwidth', -1)

top[['index','abstract',]].style.set_properties(**{'text-align': "justify"})
pd.set_option('display.max_colwidth', -1)

top[['index','url']].style.set_properties(**{'text-align': "left"})