# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns                       #visualisation

import matplotlib.pyplot as plt             #visualisation

df = pd.read_csv("../input/comcast-telecom-consumer-complaints/Comcast Telecom Complaints data.csv")

df.head()
df.info()
df["Customer Complaint"]= df["Customer Complaint"].str.lower()

df.head()
cc = list(df["Customer Complaint"])
def converttostr(input_seq, seperator):

   # Join all the strings in list

   final_str = seperator.join(input_seq)

   return final_str

seperator = ' '

cd = converttostr(cc, seperator)
import nltk

from nltk.tokenize import word_tokenize

word_tk = word_tokenize(cd)

print("word tokenizing the text: \n")

print(word_tk)
from nltk.corpus import stopwords

sw = nltk.corpus.stopwords.words('english')

newStopWords = ["comcast", ",", ".", "-", "&"]

for i in newStopWords:

    sw.append(i)

print(sw)
filtered_words = [w for w in word_tk if not w in sw]

print (filtered_words)
from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize



port_stem = PorterStemmer()
stemmed_words = []



for w in filtered_words:

    stemmed_words.append(port_stem.stem(w))



print(stemmed_words)
from nltk.probability import FreqDist

fd = FreqDist(stemmed_words)

print(fd)
fd.plot(30, cumulative = False)
from PIL import Image # if you don't have it, you'll need to install it

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
stopwords = set(STOPWORDS)

stopwords.update(["comcast", ",", ".", "-", "&"])

cloud = WordCloud(stopwords=stopwords, background_color='white').generate(cd)





# Display the generated image:

plt.figure(figsize=(13, 10), facecolor=None) 

plt.imshow(cloud, interpolation='bilinear')

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
df1 = df.drop(['Customer Complaint'], axis=1)
df1["Status"] = df1["Status"].replace('Solved', 'Closed')

df1["Status"] = df1["Status"].replace('Pending', 'Open')
df1.head(10)