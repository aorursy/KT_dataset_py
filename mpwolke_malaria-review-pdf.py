#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS51RsLc3SVYQiJs7ajWMf0rRFPVhPNB59eEDKaGfiNNzOr8gff&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
# Look at page 4

pdf = os.path.join(outputFolder,'Malaria_review.pdf[4]')

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/Malaria_review4.jpg') # intro page to preview later
# Parse a PDF file and convert it to CSV using PyTesseract

import pytesseract

pdfimage = Image.open('/kaggle/working/Malaria_review4.jpg')

text = pytesseract.image_to_string(pdfimage)  

df = pd.DataFrame([text.split('\n')])
# Plot WordCloud of page 4

plt.figure(figsize=(10,10))

wordCloudFunction(df.T,0,10000000)

plt.figure(figsize=(10,10))

wordBarGraphFunction(df.T,0,"Most Common Words on Page 4 of Malaria Review")
# Parse a PDF file and convert it to CSV using Tika-Python

!pip install tika

import tika

from tika import parser

tika.initVM()

parsed = parser.from_file('/kaggle/working/Malaria_review4.jpg') 

text = parsed["content"]

df = pd.DataFrame([text.split('\n')])

df.drop(df.iloc[:, 1:46], inplace=True, axis=1)
# Convert PDF to JPG and then convert JPG to CSV

# I will do this for Pages 289 to 291 but

# Eventually I should loop through the entire document



# PDF to JPG for p4

pdf = os.path.join(outputFolder,'Malaria_review.pdf[4]')

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/Malaria_review4.jpg')

pdfimage4 = Image.open('/kaggle/working/Malaria_review4.jpg')
# PDF to JPG for p12

pdf = os.path.join(outputFolder,'Malaria_review.pdf[12]')

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/Malaria_review12.jpg')

pdfimage12 = Image.open('/kaggle/working/Malaria_review12.jpg')



# PDF to JPG for p2

pdf = os.path.join(outputFolder,'Malaria_review.pdf[2]')

with Img(filename=pdf, resolution=300) as img:

    img.compression_quality = 99

    img.convert("RGBA").save(filename='/kaggle/working/Malaria_review2.jpg')

pdfimage2 = Image.open('/kaggle/working/Malaria_review2.jpg')
# Parse a PDF file and convert it to CSV using PyTesseract (p4)

text = pytesseract.image_to_string(pdfimage4)

df = pd.DataFrame([text.split('\n')])

df.drop(df.iloc[:, 27:], inplace=True, axis=1)

df.drop(df.iloc[:, :3], inplace=True, axis=1)

df.columns = range(df.shape[1])
# Parse a PDF file and convert it to CSV using Tika-Python (p290-291)

tika.initVM()

parsed = parser.from_file('/kaggle/working/Malaria_review12.jpg')

parsed2 = parser.from_file('/kaggle/working/Malaria_review2.jpg')



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
#Explore page 4 - Mueller Report. Here I don't know how many pages each Cheat Sheet. There are 20 pages 

w, h = pdfimage.size # crop image

pdfimage4.crop((0, 1240, w, h-1300)) # display exerpt of PDF
#Explore page 12 - Mueller Report. Here I don't know how many pages each Cheat Sheet. There are 30 pages 

w, h = pdfimage12.size # crop image

pdfimage12.crop((0, 1240, w, h-1300)) # display exerpt of PDF
#Explore page 2 - Mueller Report. Here I don't know how many pages each Cheat Sheet. There are 30 pages 

w, h = pdfimage2.size # crop image

pdfimage2.crop((0, 1240, w, h-1300)) # display exerpt of PDF
# Pages 4, 12 and 2

dfcombined.head() # preview csv of 289-291
# Clean up the notebook

!apt-get install zip # install zip

!zip -r pdfs.zip /kaggle/working/pdfs/ # zip up a few files

!rm -rf pdfs/* # remove everything else