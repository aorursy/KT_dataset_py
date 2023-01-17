from lxml import html

import numpy as np

import requests

#tree = html.fromstring(page.content)

import nltk

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk import pos_tag

import math

import re

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import spacy
import urllib.request

from bs4 import BeautifulSoup
URL1 = "https://www.essex.ac.uk/departments/computer-science-and-electronic-engineering"

URL2 = 'http://csee.essex.ac.uk/staff/udo/index.html'
#doing html parsing by use of beautiful soup all the main body parts of the Web page which will extract with the meta tags 

#and attributes of page like paragraph p and list li 

#after extracting all the html tags and body part of web page all the information of the web page in form of line and cforming of chunks

def get_html_parsin(url_):

    html = urllib.request.urlopen(url_).read()

    soup = BeautifulSoup(html)



    # kill all script and style elements

    for script in soup(["script", "style"]):

        script.extract()    # rip it out



    # get text

    text = soup.get_text()



    # break into lines and remove leading and trailing space on each

    lines = (line.strip() for line in text.splitlines())

    # break multi-headlines into a line each

    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # drop blank lines

    text = ' '.join(chunk for chunk in chunks if chunk)

    text=text.replace('\\','')

    return text
#ss= get_html_parsin(URL1)



def remove_string_special_characters(s):

    stripped = re.sub('[^\w\s]','',s)

    stripped = re.sub('_','',stripped)

    stripped = re.sub('\s',' ',stripped)

    stripped = stripped.strip()

    return stripped



#sa = remove_string_special_characters(ss)

#sa
# removing all the stop words and stemming of the word is done using library post stremer library of nltk

def get_texting_clean(data):

    data = word_tokenize(data)

    stop_words = set(stopwords.words('english'))

    clean_text = []

    ps = PorterStemmer()

    for i in data:

        if i not in stop_words:

            clean_text.append(ps.stem(i))

    return clean_text

        
sample1= get_html_parsin(URL1)

c1 = get_texting_clean(sample1)

c1 = ' '.join(w for w in c1)





sample2 = get_html_parsin(URL2)

c2 = get_texting_clean(sample2)

c2 = ' '.join(w for w in c2)
# COMBINING BOTH WEBSITES INFORMATION TO PERFORM TFIDF RANKINGS

corpus = [c1, c2]
with open('cleaned_text_of_1.txt', 'w') as f:

    f.write(c1)
# TOKENIZING OF THE WORDS AND TOKENIZING THEM ACCORDING TO PARTS OF SPEECH USING NLTK LIBRARY POS_TAG

def tagging(text):

    words = word_tokenize(text)

    #print(pos_tag(words))

    return (pos_tag(words))
tags = tagging(c1)

#tags
with open('link1_tags.txt', 'w') as output:

    for item in tags:

        output.write("{}\t".format(item))
#ss= get_html_parsin(URL2)

#cleand2 = get_texting_clean(ss)
with open('cleaned_text_of_2.txt', 'w') as f:

    f.write(c2)

   
def tagging(text):

    words = word_tokenize(text)

    #print(pos_tag(words))

    return (pos_tag(words))
#tags = tagging(cleand2)
with open('link2_tags.txt', 'w') as output:

    for item in tags:

        output.write("{}\t".format(item))
#@corpus = [cleand1, cleand2]
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(corpus)
#PERFORMING OF TFIDF RANKING AND SORTING THEM ACCORDINGLY 

with open('selectedwords.txt','w') as output:

    for i in range(0,2):

        feature_array = np.array(vectorizer.get_feature_names())

        tfidf_sorting = np.argsort(X[i].toarray()).flatten()[::-1]

        n = 30

        top_n = feature_array[tfidf_sorting][:n]

        print(top_n)

        output.write("{}\t".format(top_n))

        print(

            "\n")

        

    

    

    



    

    

    
# USING LIBRARY SPACY AND PERFORMING OF THE ENTITY AND RELATION TAGS ACCORDING TO THE WORDS 

nlp = spacy.load('en_core_web_sm')



document1 = nlp(sample1)

entity_list =[]

for ent in document1.ents:

    entity_list.append([ent, ent.label_])
entity_list
document2 = nlp(sample2)

entity_list1 =[]

for ent in document2.ents:

    entity_list1.append([ent, ent.label_])
entity_list1
with open('NER2.txt', 'w') as output:

    for item in entity_list1:

        output.write("{}\t".format(item))
with open('NER1.txt', 'w') as output:

    for item in entity_list:

        output.write("{}\t".format(item))