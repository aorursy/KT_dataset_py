import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
for i in range(1,3):
    try_url = "https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=HSL01&scat=&pageno="+str(i)+"&next=0&durationType=Y&Year=2020&duration=1&news_type="
    print(try_url)
from time import sleep
url_dict_hdfc = {}
content = []
for i in range(1,3):
    try_url = "https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=HSL01&scat=&pageno="+str(i)+"&next=0&durationType=Y&Year=2020&duration=1&news_type="
    response = requests.get(try_url)
    sleep(1)
    if response.status_code == 200:
        results_page = BeautifulSoup(response.content,'lxml')
        div_flr = results_page.findAll('div',attrs={'class':"MT15 PT10 PB10"})
        links = []
        for link in div_flr:
            url_all = link.find("a")
            if url_all != None:
                links.append(url_all.get("href"))
        for i in links:
            url_new = "https://www.moneycontrol.com" + i
            response_new = requests.get(url_new)
            results_page_new = BeautifulSoup(response_new.content,'lxml')
            table =  results_page_new.find('div',attrs={"class":"arti-flow"})
            p_article = table.findAll("p")
            for i in p_article:
                content.append(i.get_text())
            title = results_page_new.find('h1',class_="artTitle").getText()
            time_publish = results_page_new.find('div',class_="arttidate_pub").getText()
            url_dict_hdfc[url_new] = title, content, time_publish
            content = []
            
    else:
        print("Failure")

df_news_hdfc = pd.DataFrame.from_dict(url_dict_hdfc, orient = "index").reset_index()
df_news_hdfc = df_news_hdfc.rename({"index" : "URL", 0 : "Title", 1 : "Text", 2 : "Time"}, axis = 1)
df_news_hdfc["Text"] = df_news_hdfc["Text"].apply(', '.join)
df_news_hdfc
links
for i in range(1,3):
    try_url = "https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=IPL01&scat=&pageno="+str(i)+"&next=0&durationType=Y&Year=2020&duration=1&news_type="
    print(try_url)
url_dict_icici = {}
content = []
for i in range(1,3):
    try_url = "https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=IPL01&scat=&pageno="+str(i)+"&next=0&durationType=Y&Year=2020&duration=1&news_type="
    response = requests.get(try_url)
    sleep(1)
    if response.status_code == 200:
        results_page = BeautifulSoup(response.content,'lxml')
        div_flr = results_page.findAll('div',attrs={'class':"MT15 PT10 PB10"})
        links = []
        for link in div_flr:
            url_all = link.find("a")
            if url_all != None:
                links.append(url_all.get("href"))
        for i in links:
            url_new = "https://www.moneycontrol.com" + i
            response_new = requests.get(url_new)
            results_page_new = BeautifulSoup(response_new.content,'lxml')
            table =  results_page_new.find('div',attrs={"class":"arti-flow"})
            if(table == None):
                continue
            else:
                p_article = table.findAll("p")
                #print(p_article)
                #print("\n" + "\n")
                for i in p_article:
                    content.append(i.get_text())
                title = results_page_new.find('h1',class_="artTitle").getText()
                time_publish = results_page_new.find('div',class_="arttidate_pub").getText()
                url_dict_icici[url_new] = title, content, time_publish
                content = []
            
            
    else:
        print("Failure")

df_news_icici = pd.DataFrame.from_dict(url_dict_icici, orient = "index").reset_index()
df_news_icici = df_news_icici.rename({"index" : "URL", 0 : "Title", 1 : "Text", 2 : "Time"}, axis = 1)
df_news_icici["Text"] = df_news_icici["Text"].apply(', '.join)
df_news_icici
links
url_dict_bajaj = {}
content = []


import re

'''
for tag in soup.find_all(re.compile("^value_xxx_c_1_f_8_a_")):
    print(tag.name)
'''

try_url = "https://www.moneycontrol.com/news/tags/bajaj-allianz-life-insurance.html/news/"
response = requests.get(try_url)
sleep(1)
if response.status_code == 200:
    results_page = BeautifulSoup(response.content,'lxml')
    div_flr = results_page.findAll('li',attrs={'id': re.compile("^newslist-")})
    links = []
    for link in div_flr:
        url_all = link.find("a")
        if url_all != None:
            links.append(url_all.get("href"))
    
    for i in links:
        url_new = i
        response_new = requests.get(url_new)
        results_page_new = BeautifulSoup(response_new.content,'lxml')
        table =  results_page_new.find('div',attrs={"class":"arti-flow"})
        if(table == None):
            continue
        else:
            p_article = table.findAll("p")
            #print(p_article)
            #print("\n" + "\n")
            for i in p_article:
                content.append(i.get_text())
            title = results_page_new.find('h1',class_="artTitle").getText()
            time_publish = results_page_new.find('div',class_="arttidate_pub").getText()
            url_dict_bajaj[url_new] = title, content, time_publish
            content = []
                    
else:
    print("Failure")

df_news_bajaj = pd.DataFrame.from_dict(url_dict_bajaj, orient = "index").reset_index()
df_news_bajaj = df_news_bajaj.rename({"index" : "URL", 0 : "Title", 1 : "Text", 2 : "Time"}, axis = 1)
df_news_bajaj["Text"] = df_news_bajaj["Text"].apply(', '.join)
df_news_bajaj
links
import nltk
from nltk import FreqDist
nltk.download('stopwords')
import numpy as np
import re
import spacy

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# function to plot most frequent terms
def freq_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()
freq_words(df_news_hdfc["Text"])
import string
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

i_stp = ["per","rs","life","insurance","said", "cent","ltd","r"]
for i in i_stp:
    stop_words.append(i)

df_news_hdfc.Text = df_news_hdfc.Text.apply(lambda x: x.lower())
df_news_hdfc.Text = df_news_hdfc.Text.apply(lambda x: x.translate(str.maketrans("","", string.punctuation)))
df_news_hdfc.Text = df_news_hdfc.Text.apply(lambda x: x.translate(str.maketrans("","", string.digits)))

def remove_stopwords(text):    
    clean_text = " ".join([i for i in text if i not in stop_words])
    return clean_text

clean_news = [remove_stopwords(r.split()) for r in df_news_hdfc["Text"]]
freq_words(clean_news)
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output
tokenized_news = pd.Series(clean_news).apply(lambda x: x.split())
print(tokenized_news[1])
clean_news_2 = lemmatization(tokenized_news)
print(clean_news_2[1]) 
clean_news_3 = []
df = pd.DataFrame()
for i in range(len(clean_news_2)):
    clean_news_3.append(' '.join(clean_news_2[i]))

df["clean_news"] = clean_news_3

freq_words(df["clean_news"], 35)
dictionary = corpora.Dictionary(clean_news_2)
doc_term_matrix = [dictionary.doc2bow(news) for news in clean_news_2]
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100,
                chunksize=1000, passes=50)
lda_model.print_topics()
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis
freq_words(df_news_hdfc["Title"])
import string
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

i_stp = ["per","rs","life","insurance","said", "cent","ltd","r"]
for i in i_stp:
    stop_words.append(i)

df_news_hdfc.Title = df_news_hdfc.Title.apply(lambda x: x.lower())
df_news_hdfc.Title = df_news_hdfc.Title.apply(lambda x: x.translate(str.maketrans("","", string.punctuation)))
df_news_hdfc.Title = df_news_hdfc.Title.apply(lambda x: x.translate(str.maketrans("","", string.digits)))

def remove_stopwords(text):    
    clean_text = " ".join([i for i in text if i not in stop_words])
    return clean_text

clean_news = [remove_stopwords(r.split()) for r in df_news_hdfc["Title"]]
freq_words(clean_news)
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output
tokenized_news = pd.Series(clean_news).apply(lambda x: x.split())
print(tokenized_news[1])
clean_news_2 = lemmatization(tokenized_news)
print(clean_news_2[1]) 
clean_news_3 = []
df = pd.DataFrame()
for i in range(len(clean_news_2)):
    clean_news_3.append(' '.join(clean_news_2[i]))

df["clean_news"] = clean_news_3

freq_words(df["clean_news"], 35)
dictionary = corpora.Dictionary(clean_news_2)
doc_term_matrix = [dictionary.doc2bow(news) for news in clean_news_2]
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100,
                chunksize=1000, passes=50)
lda_model.print_topics()
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis