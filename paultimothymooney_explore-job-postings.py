# Import python packages

import os

from os import walk

import shutil

from shutil import copytree, ignore_patterns

from PIL import Image

from wand.image import Image as Img

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS



# Define helper function for plotting word clouds

def wordCloudFunction(df,column,numWords):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=numWords,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



# Define helper function for plotting word bar graphs

def wordBarGraphFunction(df,column,title):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])

    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))

    plt.title(title)

    plt.show()



# Preview the data folder

inputFolder = '../input/cityofla/CityofLA/Additional data/PDFs'

for root, directories, filenames in os.walk(inputFolder):

    for filename in filenames: 

        print(os.path.join(root,filename))

        

# Reorganize the data folder

outputFolder = '/kaggle/working/pdfs/'

shutil.copytree(inputFolder,outputFolder,ignore=ignore_patterns('*.db'))

for root, directories, filenames in os.walk(outputFolder, topdown=False):

    for file in filenames:

        try:

            shutil.move(os.path.join(root, file), outputFolder)

        except OSError:

            pass

shutil.rmtree(os.path.join(outputFolder,'2018'),

              os.path.join(outputFolder,'2017'),

              os.path.join(outputFolder,'2016'))

shutil.rmtree(os.path.join(outputFolder,'2015'),

              os.path.join(outputFolder,'2014'))

print(os.listdir(outputFolder))
files = next(os.walk('/kaggle/working/pdfs/'))[2] 

print("Total Number of Files in CityofLA Dataset: ", len(files))
# Preview a PDF file

pdf = os.path.join(outputFolder,'APPRENTICE - METAL TRADES 3789.pdf')

Img(filename=pdf, resolution=300)
# Parse a PDF file

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/test_jpg.jpg')

!pip install tika

import tika

from tika import parser

tika.initVM()

parsed = parser.from_file('/kaggle/working/test_jpg-1.jpg')

text = parsed["content"]

df = pd.DataFrame([text.split('\n')])

df.drop(df.iloc[:, 1:46], inplace=True, axis=1)
# Make a Word Cloud

plt.figure(figsize=(15,15))

wordCloudFunction(df.T,0,10000000)



plt.figure(figsize=(10,10))

wordBarGraphFunction(df.T,0,"Most Common Words in Job Posting")
# Preview a Text File

job_posting = '../input/cityofla/CityofLA/Job Bulletins/APPRENTICE - METAL TRADES 3789 070816.txt'

with open(job_posting) as f: 

    print (f.read(1000))
# Clean Up The Notebook

!apt-get install zip

!zip -r pdfs.zip /kaggle/working/pdfs/

!rm -rf pdfs/* 