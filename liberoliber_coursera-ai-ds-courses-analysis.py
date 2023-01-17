import warnings

warnings.filterwarnings("ignore")



import pandas as pd

from pandas import ExcelWriter

from pandas import ExcelFile



import numpy as np

import nltk

import re

import datetime

import matplotlib.pyplot as plt

# %% [code]

FIGSIZE = (16, 9)

FONT = {"family": "Share Tech Mono", "weight": "normal", "size": 16}

tds = "#0073F1"

week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]



#UrlLib for http handlings

import urllib.request  

import bs4 as BeautifulSoup

import nltk



#from string import punctuation

#from nltk.corpus import stopwords

#from nltk.tokenize import word_tokenize

#from nltk.tokenize import sent_tokenize



from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import urllib.request  

import bs4 as BeautifulSoup

import nltk

from string import punctuation



#For the next version 

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



!pip install pyshorteners

#!pip install xlrd



import pyshorteners
df = pd.read_excel('/kaggle/input/ai-ds-coursera-trainings/CS_DS_Free_and_FreeToAudit_Jair_Ribeiro.xlsx')
courses = df

courses
def summarization_courses(i):

    # Data collection from the links using web scraping(using Urllib library)

    course_url = courses['URL'].tolist()

    course_url = course_url[i]

    #course_url = ""





    text = urllib.request.urlopen(course_url)

    text



    course_summary = text.read()

    course_summary

    

    # Parsing the URL content 

    course_parsed = BeautifulSoup.BeautifulSoup(course_summary,'html.parser')

    

    # Returning <p> tags

    paragraphs = course_parsed.find_all('p')

    

    # To get the content within all poaragrphs loop through it

    course_content = ''

    for p in paragraphs:  

        course_content += p.text

    

    

    # 3) Tokenization & Data clean up    

    tokens = word_tokenize(course_content)

    stop_words = []

    stopwords  = []

    punctuation = " "

    stopwords = set(STOPWORDS)

    punctuation = punctuation + '\n'

    punctuation

    

    word_frequencies = {}

    for word in tokens:    

        if word.lower() not in stop_words:

            if word.lower() not in punctuation:

                if word not in word_frequencies.keys():

                    word_frequencies[word] = 1

                else:

                    word_frequencies[word] += 1

    #word_frequencies

    

    max_frequency = max(word_frequencies.values())

    #print(max_frequency)

    

    for word in word_frequencies.keys():

        word_frequencies[word] = word_frequencies[word]/max_frequency

       #print(word_frequencies)

    

    sent_token = sent_tokenize(course_content)

    #print(sent_token)

    

    

    sentence_scores = {}

    for sent in sent_token:

        sentence = sent.split(" ")

        for word in sentence:        

            if word.lower() in word_frequencies.keys():

                if sent not in sentence_scores.keys():

                    sentence_scores[sent] = word_frequencies[word.lower()]

                else:

                    sentence_scores[sent] += word_frequencies[word.lower()]

    #sentence_scores

    

    from heapq import nlargest

    select_length = int(len(sent_token)*0.20) # Set the summary to 8% of the whole article.

    select_length = 1 # Set the summary to 1 sentence.

    

    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)

    final_summary = [word for word in summary]

    summary = ' '.join(final_summary)

    

    return summary

s = pyshorteners.Shortener()

print(s.tinyurl.short(df["URL"][2])) # just checking if it works!
courses.head()
print("Summary of the AI & Data Science Courses")

print()

i=0

n = int(courses["URL"].count())

for i in range(n):

    print("-------------------------------------")

            

    

    type_of_training = courses["Type"][i]

    if type_of_training == "Free":

        print(" ")

        print(courses["Course Title"][i] + " (Free)")

        

    else:

        print(" ")

        print(courses["Course Title"][i] + " (Free to Audit)")

    

    print(" ")

    print(summarization_courses(i))



    tiny = str(courses["URL"][i])

    print(" ")

    print("University: " + courses["University Name"][i])

    print("Domain: " + courses["Subject Subdomain"][i])

    print("Rating: " + str(courses["Rating"][i]))

    print("Link: "+ s.tinyurl.short(tiny))

    print(" ")

    

    i=i+1
df
#Connections Plotting Functions





def plot_bar_column(courses, col):

    fnames = courses[col].value_counts().head(30)

    plot_fnames(fnames,col)





def plot_nlp_cv(courses):

    tfidf = CountVectorizer(ngram_range=(1, 3), stop_words='english')

    cleaned_positions = list(courses["Course Title"].fillna(""))

    res = tfidf.fit_transform(cleaned_positions)

    res = res.toarray().sum(axis=0)



    fnames = pd.DataFrame(

        list(sorted(zip(res, tfidf.get_feature_names())))[-30:],

        columns=["Course Title by Words Freq", "Words"]

    )[::-1] 

    plot_fnames(fnames, "Course Title by Words Freq", "Words")





def plot_fnames(fnames, col, index="index"):

    fnames = fnames.reset_index()



    fig, ax = plt.subplots(figsize=FIGSIZE)



    plt.bar(

        x=fnames.index,

        height=fnames[col],

        color=tds,

        alpha=0.5

    )



    plt.title("{} distribution".format(col), fontdict=FONT, y=1.2)

    plt.xticks(

        fnames.index,

        fnames[index].str.capitalize(),

        rotation=65,

        ha="right",

        size=FONT["size"],

    )



    plt.ylabel("Nb occurences", fontdict=FONT)

    plt.yticks()#[0, 5, 10, 15, 20])

    ax.set_frame_on(False)

    plt.grid(True)



    plt.show()
courses.head()
plot_nlp_cv(courses)
plot_bar_column(courses, "University Name")
plot_bar_column(courses, "Subject Subdomain")
plot_bar_column(courses, "Type")
#Top 10 Universities

universities = courses['University Name'].value_counts().head(10)

universities



ax = universities.plot(kind='bar',

                                    figsize=(14,8),

                                    title="University providing trainings")

ax.set_xlabel("University")

ax.set_ylabel("Number of Courses")
#Top 10 Domains

domains = courses['Subject Subdomain'].value_counts().head(10)

domains





ax = domains.plot(kind='bar',

                                    figsize=(14,8),

                                    title="Top 10 Domains")

ax.set_xlabel("Domains")

ax.set_ylabel("Number of courses")
courses.head()
courses['Subject Subdomain'].value_counts().plot(kind='bar');