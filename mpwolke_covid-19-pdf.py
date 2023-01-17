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
# Import Python Packages

# PyTesseract and Tika-Python for OCR

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import shutil

import PIL

import os

from os import walk

from shutil import copytree, ignore_patterns

from collections import Counter

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

from PIL import Image

from wand.image import Image as Img

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

#mueller_report = pd.read_csv('../input/data-science-cheat-sheets/Interview Questions/AI Questions.pdf') # one row per line
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

inputFolder = '../input/'

for root, directories, filenames in os.walk(inputFolder):

    for filename in filenames: 

        print(os.path.join(root,filename))

        

# Move data to folder with read/write access

outputFolder = '/kaggle/working/pdfs/'

shutil.copytree(inputFolder,outputFolder,ignore=ignore_patterns('*.db'))

for root, directories, filenames in os.walk(outputFolder, topdown=False):

    for file in filenames:

        try:

            shutil.move(os.path.join(root, file), outputFolder)

        except OSError:

            pass

print(os.listdir(outputFolder))
# Look at intro page

pdf = os.path.join(outputFolder,'20200224-sitrep-35-covid-19.pdf[1]')

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/20200224-sitrep-35-covid-19.jpg') # intro page to preview later
# Parse a PDF file and convert it to CSV using PyTesseract

import pytesseract

pdfimage = Image.open('/kaggle/working/20200224-sitrep-35-covid-19.jpg')

text = pytesseract.image_to_string(pdfimage)  

df = pd.DataFrame([text.split('\n')])
# Plot WordCloud of page 1

plt.figure(figsize=(15,15))

wordCloudFunction(df.T,0,10000000)

plt.figure(figsize=(10,10))

wordBarGraphFunction(df.T,0,"Most Common Words on Page 1 of the 20200224-sitrep-35-covid-19") 