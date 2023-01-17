# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 

import numpy as np 

import os 

import pandas as pd 

from datetime import datetime
df = pd.read_csv('/kaggle/input/build-a-tool-to-deduce-skill-set/developer_india_5471_20200214_1581673429402490_1.csv')

data = df.head()
df.shape
df.info()
df['crawl_timestamp'] = pd.to_datetime(df['crawl_timestamp'])
df['crawl_timestamp'].head(5)
df["job_title"].value_counts()

df["job_title"].value_counts().head().plot.bar(figsize=(20,6))
df["category"].value_counts().head(10)
df['company_name'].value_counts().head().plot(kind = 'bar')
df['category'].value_counts().head(10).plot(kind = 'bar')
df["html_job_description"].isnull().sum()
df['job_title'].value_counts().head(30).plot(kind = 'bar')

requirements = {"support, maintenance":0,"web application":0,"mysql":0,"SAS Developer":0,".Dotnet Developer Fresher":0,"MS Word":0,"PHP Developer":0, " r ":0, "C":0, "Data Structure":0, "python":0, "PL SQL":0, "machine learning":0,'linux':0, 'c#':0, \

" ml ":0, "Asp.Net MVC":0, "spark":0, "hadoop":0, "java":0, "scala":0, "HTML":0, "CSS":0, "JavaScript":0, "React ":0, 'jQuery':0, 'Node.js':0, "c++":0}

for i in range(len(df)):

    html_job_description = str(df.html_job_description[i]).lower().replace("\n", " ")

    for k in requirements:

        if k in html_job_description:

            requirements[k] += 1

print(requirements['machine learning'])

print(requirements[' ml '])

requirements['machine learning'] += requirements[' ml ']
from collections import OrderedDict

sorted_req = OrderedDict(sorted(requirements.items(), key=lambda x:x[1]))

plt.figure(figsize=(10, 10))

plt.bar(range(len(sorted_req)), list(sorted_req.values()), align='center')

plt.xticks(range(len(sorted_req)), list(sorted_req.keys()), rotation='vertical')

plt.xlabel("job requirement")

plt.ylabel("Number of posts")

plt.show()
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

description_example = df.html_job_description[2].lower()

word_tokens = word_tokenize(description_example)

filtered_description = [w for w in word_tokens if not w in stop_words]

filtered_description = " ".join(filtered_description)
from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import warnings

warnings.filterwarnings("ignore")
aggregate_descriptions = " ".join(str(job_description)

                      for job_description in df.html_job_description)

stopwords = set(STOPWORDS)



wordcloud = WordCloud(stopwords=stopwords, background_color='white',

                     width=1000, height=700).generate(aggregate_descriptions)

plt.figure(figsize=(15, 10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
import re

from word2number import w2n

import statistics

def search_text_left_of_word(text, word, n):

    

    words = re.findall(r'\w+', text)

    try:

        index = words.index(word)

    except ValueError:

        return " "

    return words[index - n:index]

def search_year_word(text):

    return text.find('year')

def search_number_around_word(word_surroundings):          #this function adds all the numbers found to a list. It also converts the words to numbers, if it is the case

    word_surroundings = " ".join(word_surroundings)

    word_surroundings = word_tokenize(word_surroundings)

    pos_tags = nltk.pos_tag(word_surroundings)

    numbers_list = []

    for a in pos_tags:

        if a[1] in 'CD':

            if a[0].isalpha():       #sometimes the numbers are written as words, e.g. 'Three' instead of 3

                try:

                    numbers_list.append(w2n.word_to_num(a[0]))

                except ValueError:

                    return ""

            else:

                numbers_list.append(a[0])

    return numbers_list

years_experience_req = []
def convert_to_int(list_elem):

    try:

        converted_int = int(list_elem)

        if converted_int <= 10:

            return int(list_elem)

    except ValueError:

        return

for post_index in range (len(df)):

    current_job = str(df.html_job_description[post_index])

    word_surroundings = search_text_left_of_word(current_job, 'years', 2)

    if current_job.find(' year ') > -1:

        years_experience_req.append(['1'])

    years_experience_req.append(search_number_around_word(word_surroundings))

    #print(post_index, search_number_around_word(word_surroundings))

years_experience_req = [convert_to_int(item) for sublist in years_experience_req for item in sublist]

years_experience_req = [i for i in years_experience_req if i != None]

#print(years_experience_req)

print("An average of ", statistics.mean(years_experience_req), "  years is required in most job offerings. ")

df.html_job_description[10]