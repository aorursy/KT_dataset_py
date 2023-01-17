#!pip install --upgrade pip

!pip install transformers

!pip install torch

!pip install pyshorteners

#!pip install xlrd

#!pip install wikipedia

!pip install vaderSentiment
import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np



import urllib.request

import re

    

    

import datetime



#UrlLib for http handlings

from   bs4 import BeautifulSoup

import bs4 as BeautifulSoup

import urllib.request

from urllib.request import urlopen 

from socket import timeout





#WordCloud - not used in this version.

#from os import path

#from PIL import Image

#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



#For the NLP  

import nltk

from string import punctuation

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



#For the Tf 

import tensorflow as tf

import torch

import json 

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, pipeline

from transformers import TransfoXLTokenizer, TransfoXLModel

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import torch

#Wikipedia

#import wikipedia



#TinyUrl

import pyshorteners



import gc



import time
start_time = time.time()
links = pd.read_excel('../input/newsletterdata/NewsLetter_Links.xlsx',encoding='Latin–1')

#links    = pd.read_csv("../input/SRWPwAI.csv",encoding='utf8' )

links['to_summarize'] = 0

links['article_text'] = ""

links
#Sorting by date

links=links.sort_values(by="Title")
links
links = links[links['Url'].notna()]



links.shape
# Resetting the dataframe index

links.reset_index(inplace=True)
# Dropping the Index and creating a column called "number" as index

links.drop(columns="index",inplace=True)

links.rename(columns={"level_0":"number"},inplace=True)
links.dropna(inplace=True)

links.drop_duplicates(subset ="Url",keep = False, inplace = True) 

links.shape
#final version of the Dataframe

links
# Resetting the dataframe index

links.reset_index(inplace=True)



# Dropping the Index and creating a column called "number" as index

links.drop(columns="index",inplace=True)

links.rename(columns={"level_0":"number"},inplace=True)
links.shape
links = links[links['Url'].notna()]

links.shape
links = links[links['Title'].notna()]

links.shape
links.dropna(inplace=True)

#links.drop_duplicates(subset ="MediaURL",keep = False, inplace = True) 

links.drop_duplicates(subset ="Url",keep = False, inplace = True) 

links.shape
# Resetting the dataframe index

links.reset_index(inplace=True)
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"

my_headers = {'User-Agent': user_agent, 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}



values = {'name': 'Michael Foord',

          'location': 'Northampton',

          'language': 'Sweden' }
# import module sys to get the type of exception

import sys

import urllib.request

from urllib.request import urlopen 

from socket import timeout

from urllib.request import build_opener, HTTPCookieProcessor, Request



n = int(links["Url"].count())

n =3 # to keep this short I reduce the number of articles to 3 - remove this limitation if you want to summarize everything.



print("Checking " + str(n)+" article's links for errors accessing the source websites...")

print()

i=0



for i in range(n):

    

    print("Checking access to article number: "+ str(i) + " - " + str(links["Title"][i]) + ".")

    req = urllib.request.Request(links["Url"][i], headers=my_headers)

    try: 

        urllib.request.urlopen(req,timeout=1000)

        print("Access OK")

        links['to_summarize'][i] = 1

    except Exception as e:

        print("Oops!", e.__class__, "occurred.")

        print()

print("All the Links were verified.")
links
import random

from random import seed

from random import sample

from numpy.random import shuffle

# seed random number generator

seed(1)

# prepare a sequence

sequence = [i for i in range(n)]

#print(sequence)

shuffle(sequence)

# select a subset without replacement

subset = sample(sequence, 1)

#print(subset)





numberList = subset

print("random article to be tested: ", random.choice(numberList))



url_test = random.choice(numberList)

#url_test
d = links

i = url_test
import pyshorteners



s = pyshorteners.Shortener()

print(s.tinyurl.short(links["Url"][url_test]))
def text_extract(d,i):

    

    # Data collection from the links using web scraping(using Urllib library)

    links_url = d['Url'].tolist()

    links_url = links_url[i]

    #links_url = test_url



    req = urllib.request.Request(links["Url"][i], headers=my_headers)

    text = urllib.request.urlopen(req, timeout=100)

    

    summary = ''

    link_summary = text.read()

    

    

    # Parsing the URL content 

    link_parsed = BeautifulSoup.BeautifulSoup(link_summary,'html.parser')

    

    # Returning <p> tags

    paragraphs = link_parsed.find_all('p')

    

    # To get the content within all paragrphs loop through it

    link_content = ''

    for p in paragraphs:  

        link_content += p.text

    

    # Removing Square Brackets and Extra Spaces

    link_content = re.sub(r'\[[0-9]*\]', ' ', link_content)

    link_content = re.sub(r'\s+', ' ', link_content)

    

    # Removing special characters and digits

    formatted_article_text = re.sub('[^a-zA-Z]', ' ', link_content )

    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    

    text = formatted_article_text

    #text_temp = text.read()

    global webtext

    preprocess_text = text.strip().replace("\n","")

    #webtext = preprocess_text[0:512]

    webtext = preprocess_text

    links['article_text'][i] = webtext

    return webtext
device    = torch.device('cpu')

model     = T5ForConditionalGeneration.from_pretrained('t5-large')

tokenizer = T5Tokenizer.from_pretrained('t5-large')



#tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

#model = TFTransfoXLModel.from_pretrained('transfo-xl-wt103')





def summarization_infer(text, max=512):

  preprocess_text = text.replace("\n", " ").strip()

  t5_prepared_Text = preprocess_text

  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")



  summary_ids = model.generate(tokenized_text, min_length=100, max_length=max, top_k=100, top_p=0.8, early_stopping=False, maxfeatures=100, num_beams=3,no_repeat_ngram_size=2) #top-k top-p sampling strategy

  

  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

  #end_time = time.time()

  #print (f'Time taken : {end_time-start_time}')

  return output



def translation_infer(text, max=50):

  preprocess_text = text.replace("\n", " ").strip()

  t5_prepared_Text = "translate English to German: "+preprocess_text

  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")



  translation_ids = model.generate(tokenized_text, min_length=10, max_length=50, early_stopping=True, num_beams=2)

  output = tokenizer.decode(translation_ids[0], skip_special_tokens=True)

  #end_time = time.time()

  #print (f'Time taken : {end_time-start_time}')

  return output



def grammatical_acceptibility_infer(text):

  preprocess_text = text.replace("\n", " ").strip()

  t5_prepared_Text = "cola sentence: "+preprocess_text

  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")



  grammar_ids = model.generate(tokenized_text, min_length=1, max_length=3)

  output = tokenizer.decode(grammar_ids[0], skip_special_tokens=True)

  #end_time = time.time()

  #print (f'Time taken : {end_time-start_time}')

  return output.capitalize()



def summarization_XL(text,max_len):

  #tokenizer = TransfoXLTokenizer.from_pretrained('t5-large')

  model     = T5ForConditionalGeneration.from_pretrained('t5-large')





  #tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

  tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

  #model = TFTransfoXLModel.from_pretrained('transfo-xl-wt103')

    

  sequence_a = text

  encoded_sequence_a = tokenizer.encode(sequence_a)



  # Continuation of the previous script

  sequence_a_dict = tokenizer.encode_plus(sequence_a,  pad_to_max_length=True)



  sequence_a_dict['input_ids'] 

  sequence_a_dict['attention_mask']   

    

  input_ids = tf.constant(tokenizer.encode(text, add_special_tokens=True))[None, :]  # Batch size 1

  output = model(input_ids,attention_mask)

  last_hidden_states, mems = output[:2]

  return output
def reduce_mem_usage(d, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = d.memory_usage().sum() / 1024**2

    for col in d.columns:

        col_type = d[col].dtypes

        if col_type in numerics:

            c_min = d[col].min()

            c_max = d[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    d[col] = d[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    d[col] = d[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    d[col] = d[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    d[col] = d[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    d[col] = d[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    d[col] = d[col].astype(np.float32)

                else:

                    d[col] = d[col].astype(np.float64)

    import gc

    gc.collect()

    end_mem = d.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
reduce_mem_usage(d, verbose=True)
#summarization_pipeline = pipeline(task='summarization', model="t5-large")
def clean_summarized_text(output_text):

    

    s = str(output_text)

    s = s.replace("[{'summary_text': '", '')

    s = s.replace("'}]", '.')

    s = s.replace("na en a-ena re-a-a n aen .a oa", '')

    s = s.replace("aa na as re n a", '')

    s = s.replace("en re-a", '')

    s = s.replace("a<extra_id_27>", '')

    s = s.replace("The a aa . ", '')

    s = s.replace(" .", '.')

    s = s.replace("  .", '.')

    s = s.replace("  .  ", '.')

    

    s = s.replace("\n\n", '')

    s = s.replace(" + iWork", '')

    

    s = s.replace("[{'generated_text':", '')

     

    

    

    

    

    clean_output = s.capitalize()

    return clean_output
i=0

n = int(links["Url"].count())

#n = 2

print("Save "+str(n) +" articles to the dataframe.")

print("____________________________________________________________")

for i in range(n):

            

    

    #published = links["added"][i]

    to_summarize = links["to_summarize"][i]

    if to_summarize == 1:

            print(" ")

            print(str(i)+ ") " + links["Title"][i] + ".")

            print(" ")

            text_extract(d,i)

            

            gc.collect()

            i=i+1
links
def summarization_links_t5(d,i):

    t5_model     = T5ForConditionalGeneration.from_pretrained('t5-large')

    t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')    

    

# Data collection from the links using web scraping(using Urllib library)

    links_url = d['Url'].tolist()

    links_url = links_url[i]

    #links_url = test_url

    

    req = urllib.request.Request(links["Url"][i], headers=my_headers)

    text = urllib.request.urlopen(req)

    

    summary = ''

    link_summary = text.read()

    #link_summary = link_summary[0:512]

    

    # Parsing the URL content 

    link_parsed = BeautifulSoup.BeautifulSoup(link_summary,'lxml')

    

    # Returning <p> tags

    paragraphs = link_parsed.find_all('p')

    

    # To get the content within all paragrphs loop through it

    link_content = ''

    for p in paragraphs:  

        link_content += p.text

    

    # Removing Square Brackets and Extra Spaces

    link_content = re.sub(r'\[[0-9]*\]', ' ', link_content)

    link_content = re.sub(r'\s+', ' ', link_content)

    

    # Removing special characters and digits

    formatted_article_text = re.sub('[^a-zA-Z]', ' ', link_content )

    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    

    text = formatted_article_text

    #text_temp = text.read()



    preprocess_text = text.strip().replace("\n","")

    t5_prepared_Text = "summarize: "+preprocess_text

    #print ("original text preprocessed: \n", preprocess_text)



    tokenized_text = t5_tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)





    # summmarize 

    summary_ids = t5_model.generate(tokenized_text,

                                    num_beams=3,

                                    no_repeat_ngram_size=2,

                                    min_length=100,

                                    max_length=512,

                                    early_stopping=False,

                                    maxfeatures=10)



    global t5_output

    t5_output = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    

    #output = str(d['Title'][i]) + " - " +output 



    #print ("\n\nSummarized text: \n",output)

    

    return t5_output.capitalize()
links.head()
def summarization_links_t5_local(i):

    

    model     = T5ForConditionalGeneration.from_pretrained('t5-large')

    tokenizer = T5Tokenizer.from_pretrained('t5-large')    



    

    

    # Parsing the URL content 

    link_parsed = str(links['article_text'][i])

    

    # Returning <p> tags

    paragraphs = link_parsed.find_all('p')

    

    # To get the content within all paragrphs loop through it

    link_content = ''

    for p in paragraphs:  

        link_content += p.text

    

    # Removing Square Brackets and Extra Spaces

    link_content = re.sub(r'\[[0-9]*\]', ' ', link_content)

    link_content = re.sub(r'\s+', ' ', link_content)

    

    # Removing special characters and digits

    formatted_article_text = re.sub('[^a-zA-Z]', ' ', link_content )

    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    

    text = formatted_article_text

    #text_temp = text.read()



    preprocess_text = text.strip().replace("\n","")

    t5_prepared_Text = "summarize: "+preprocess_text

    #print ("original text preprocessed: \n", preprocess_text)



    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)





    # summmarize 

    summary_ids = model.generate(tokenized_text,

                                    num_beams=3,

                                    no_repeat_ngram_size=2,

                                    min_length=100,

                                    max_length=512,

                                    early_stopping=False,

                                    maxfeatures=10)



    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    

    #output = str(d['Title'][i]) + " - " +output 



    #print ("\n\nSummarized text: \n",output)

    return output.capitalize()
#summarization_links_t5_local(1)
#text_extract(links,url_test)
from transformers import pipeline
text = str(links["article_text"][0])

summarization_pipeline = pipeline(task='summarization', model="t5-large") 

output = summarization_pipeline(text, min_length=1, max_length=500, top_k=10, top_p=0.8)

s = str(output)

clean_summarized_text(s)
#summarization_links_t5(links,url_test)
text = str(links["article_text"][0])

#summarization_infer(text, max=512)
#grammatical_acceptibility_infer(sentiment_analyzer_scores(summarization_links_t5(d,i)))
# Extra Text Generation using Transformers based on the article's title
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):

    score = analyser.polarity_scores(sentence)

    print("{:-<40} {}".format(sentence, str(score)))
#sentiment_analyzer_scores(summarization_links_t5(d,i))
links
def check_sentiment_score(i):

    analyzer = SentimentIntensityAnalyzer()

    links['sentiment_score'] = pd.DataFrame(links.article_text.apply(analyzer.polarity_scores).tolist())['compound']

    sentiment_score = pd.cut(links['sentiment_score'], [-np.inf, -0.35, 0.35, np.inf], labels=['negative', 'neutral', 'positive'])

    links['article_sentiment'] = str(sentiment_score[i])

    return sentiment_score[i]
#i=0

#sentiment = str(check_sentiment_score(i))

#sentiment
#links
#url_test = 188

#knowledge_base = str(text_extract(links,url_test))

#knowledge_base



#webtext = preprocess_text[0:512]
def question_answer(knowledge_base):

    qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")

    qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")



    

    

    

    #knowledge_base = str(text_extract(links,i))

    text = knowledge_base[0:512]

    questions = [

        "What is this article about?",

        "Why is this article interesting?"

    ]



    for question in questions:

        inputs = qa_tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")

        input_ids = inputs["input_ids"].tolist()[0]



        text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)

        answer_start_scores, answer_end_scores = qa_model(**inputs)



        answer_start = torch.argmax(

            answer_start_scores

        )  # Get the most likely beginning of answer with the argmax of the score

        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score



        answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))



        print(f"Question: {question}")

        print(f"Answer: {answer.capitalize()}\n")
#question_answer(i)
reduce_mem_usage(d, verbose=True)
summarization_pipeline = pipeline(task='summarization', model="t5-large")
i
def new_summarizer(i):

    output =""

    text = str(links["article_text"][i])

    output = summarization_pipeline(text, num_beams=3, no_repeat_ngram_size=2, min_length=100, max_length=512, early_stopping=False,maxfeatures=10)

    s = str(output)

    summary = str(s)

    return summary
#new_summarizer(i)
#summary = str(summarization_links_t5(d,i))

#summary
i=0

summary =""

n = int(links["Url"].count())

#n = 10

print("Building the summary of "+str(n) +" articles to the newsletter.")

print("____________________________________________________________")

for i in range(n):

            

    

    #published = links["added"][i]

    to_summarize = links["to_summarize"][i]

    if to_summarize == 1:

            print(" ")

            print(links["Title"][i] + ".")

            print(" ")

            summary = str(summarization_links_t5(d,i))

            print(summary)

            #sentiment = str(check_sentiment_score(i))

            print ("")

            #print("Sentiment score of the article: " +str(sentiment))        

            question_answer(summary)

            print(" ")

            s = pyshorteners.Shortener()

            print("Link: "+ str(links["Url"][i]))

            print("____________________________________________________________")

            gc.collect()

            i=i+1

            

print("All articles were summarized! Well done!")
end_time = time.time()

print (f'Time taken to run this script: {end_time-start_time}')