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

mueller_report = pd.read_csv('../input/mueller_report.csv') # one row per line



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

pdf = os.path.join(outputFolder,'muellerreport.pdf[8]')

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/mueller8.jpg') # intro page to preview later
# Parse a PDF file and convert it to CSV using PyTesseract

import pytesseract

pdfimage = Image.open('/kaggle/working/mueller8.jpg')

text = pytesseract.image_to_string(pdfimage)  

df = pd.DataFrame([text.split('\n')])
# Parse a PDF file and convert it to CSV using Tika-Python

!pip install tika

import tika

from tika import parser

tika.initVM()

parsed = parser.from_file('/kaggle/working/mueller8.jpg') 

text = parsed["content"]

df = pd.DataFrame([text.split('\n')])

df.drop(df.iloc[:, 1:46], inplace=True, axis=1)
# Plot WordCloud of page 8

plt.figure(figsize=(15,15))

wordCloudFunction(df.T,0,10000000)

plt.figure(figsize=(10,10))

wordBarGraphFunction(df.T,0,"Most Common Words on Page 8 of the Mueller Report")
# Convert PDF to JPG and then convert JPG to CSV

# I will do this for Pages 289 to 291 but

# Eventually I should loop through the entire document



# PDF to JPG for p289

pdf = os.path.join(outputFolder,'muellerreport.pdf[289]')

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/mueller289.jpg')

pdfimage289 = Image.open('/kaggle/working/mueller289.jpg')



# PDF to JPG for p290

pdf = os.path.join(outputFolder,'muellerreport.pdf[290]')

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/mueller290.jpg')

pdfimage290 = Image.open('/kaggle/working/mueller290.jpg')



# PDF to JPG for p291

pdf = os.path.join(outputFolder,'muellerreport.pdf[291]')

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/mueller291.jpg')

pdfimage291 = Image.open('/kaggle/working/mueller291.jpg')



# Parse a PDF file and convert it to CSV using PyTesseract (p289)

text = pytesseract.image_to_string(pdfimage289)

df = pd.DataFrame([text.split('\n')])

df.drop(df.iloc[:, 27:], inplace=True, axis=1)

df.drop(df.iloc[:, :3], inplace=True, axis=1)

df.columns = range(df.shape[1])



# Parse a PDF file and convert it to CSV using Tika-Python (p290-291)

tika.initVM()

parsed = parser.from_file('/kaggle/working/mueller290.jpg')

parsed2 = parser.from_file('/kaggle/working/mueller291.jpg')



text = parsed["content"]

df2 = pd.DataFrame([text.split('\n')])

df2.drop(df2.iloc[:, 1:50], inplace=True, axis=1)

df2.drop(df2.iloc[:, 26:], inplace=True, axis=1)

df2.columns = range(df2.shape[1])



text = parsed2["content"]

df3 = pd.DataFrame([text.split('\n')])

df3.drop(df3.iloc[:, :50], inplace=True, axis=1)

df3.drop(df3.iloc[:, 22:], inplace=True, axis=1)

df3.columns = range(df3.shape[1])



dfcombined = pd.concat([df, df2, df3]) # combine pages 289-291
w, h = pdfimage289.size # crop image

pdfimage289.crop((0, 1240, w, h-1300)) # display exerpt of PDF
# Pages 289-291

dfcombined.head() # preview csv of 289-291
# Word BarGraph for Entire Document

mueller_report.drop(mueller_report.iloc[:, :2], inplace=True, axis=1) # only columns with text

plt.figure(figsize=(10,10)) # 10x10 figure

wordBarGraphFunction(mueller_report,'text',"Most Common Words in the Entire Mueller Report") # plot word bar graph
# Clean up the notebook

!apt-get install zip # install zip

!zip -r pdfs.zip /kaggle/working/pdfs/ # zip up a few files

!rm -rf pdfs/* # remove everything else