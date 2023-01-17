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
#Saved output to .txt file at this point, went through exact same process to extract text from all nights of the DNC.  
#https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-night-1-transcript
#https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-2020-night-2-transcript
#https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-night-3-transcript
#https://www.rev.com/blog/transcripts/2020-democratic-national-convention-dnc-night-4-transcript


#RNC Transcripts
#https://www.rev.com/blog/transcripts/2020-republican-national-convention-rnc-night-4-transcript
#https://www.rev.com/blog/transcripts/2020-republican-national-convention-rnc-night-2-transcript
#https://www.rev.com/blog/transcripts/2020-republican-national-convention-rnc-night-3-transcript
#https://www.rev.com/blog/transcripts/2020-republican-national-convention-rnc-night-1-transcript
#This is when we want to mess with how the html looks in the console.
import pprint

#This is the library we need to visit webpages and store their html
import requests

#Getting "BeautifulSoup" library for textual analysis.
from bs4 import BeautifulSoup

URL = 'https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-night-1-transcript'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id='transcription')
speech = results.find('div', class_='fl-callout-text')
print(speech.text)
text=speech.text
text1=(text.replace('0','').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace(':', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('Speaker', ''))
text1
import numpy as np
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import os
import random
# install wordcloud
!conda install -c conda-forge wordcloud --yes

# import package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

print ('Wordcloud is installed and imported!')
#Lets get the .txt file that has been cleaned a bit
RNC = open('../input/rnctext2/RNCConventionText.txt', 'r').read()
print ('File downloaded and saved!')
#Raw Content
print(RNC)
#We need to strip out all the timestamps and odd characters.  Using the .replace() will work...theres probably an easier way to do this.

RNC=(RNC.replace('0','').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace(':', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('Speaker', '').replace('/n', ''))
RNC

#Adding the default stop words list (things like 'and, of, the')
stopwords = set(STOPWORDS)
#Ok, thats better.  No more timestamps to get in the way. Lets make our first run for preliminary analysis.

# instantiate a word cloud object
RNC_wc = WordCloud(
    background_color='gray',
    max_words=75000,
    stopwords=stopwords
)

# generate the word cloud
RNC_wc.generate(RNC)

# Display the wordcloud
plt.imshow(RNC_wc, interpolation='bilinear')
plt.axis('off')
plt.show()
#We will need to do some processing before the final product.  We can use a 'mask' to shape the text.
#Lets create the mask image.  Works best with a .png file and a dark silhouette with a white background.
elephant = np.array(Image.open('../input/goplogos/WhitebgGOP.png'))
#Text Processing
RNC = RNC.replace(".", ". ")
RNC = RNC.replace("announcer", ".")
#RNC = RNC.replace("ve", ".")
#RNC = RNC.replace("re", ".")
#RNC = RNC.replace("ha", ".")

#lets create a grey folor function to help the words stand out.  We can change these by tweaking the parametes below.
def grey_color_func(word, font_size, position, orientation, random_state=1,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

wc = WordCloud(background_color="#d11d1d", width=1200, height=800, max_words=75000, mask=elephant, stopwords=stopwords, margin=1,random_state=2).generate(RNC)

# store default colored image
default_colors = wc.to_array()

#Custom Image size to get us a high res version.
plt.figure(figsize=(50,25))
plt.title("RNC Speeches")
plt.axis("off")
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
plt.show()



#plt.figure(figsize=(50,25))
#plt.title("Default colors")
#plt.imshow(default_colors, interpolation="bilinear")
#plt.axis("off")
#plt.show()
