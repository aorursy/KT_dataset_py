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
#This is when we want to mess with how the html looks in the console.
import pprint

#Library we need to visit webpages and store their html
import requests

#Store the name of the page (URL) to an actual url
URL = 'https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-2020-night-1-transcript'

#use requests.get
page1 = requests.get(URL)

#.content calls the page html!
page1.content
from bs4 import BeautifulSoup

URL = 'https://www.rev.com/blog/transcripts/2020-democratic-national-convention-dnc-night-4-transcript'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id='transcription')
speech = results.find('div', class_='fl-callout-text')
print(speech.text)
#print(results.prettify())
#We use .replace() to get rid of numerals and special characters with the goal of erasing the timestamps in the output.
text=speech.text
text1=(text.replace('0','').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace(':', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('Speaker', ''))
text1
#Saved output to .txt file at this point, went through exact same process to extract text from all nights of the DNC.  
#https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-night-1-transcript
#https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-2020-night-2-transcript
#https://www.rev.com/blog/transcripts/democratic-national-convention-dnc-night-3-transcript
#https://www.rev.com/blog/transcripts/2020-democratic-national-convention-dnc-night-4-transcript

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
!wget --quiet ../input/dnc-text/DNCConventionText.txt
DNC = open('/kaggle/input/dnc-text/DNCConventionText.txt', 'r').read()
print ('File downloaded and saved!')

#DNC = text4
#print("File Downloaded and saved!")
print(DNC)
stopwords = set(STOPWORDS)
# instantiate a word cloud object
DNC_wc = WordCloud(
    background_color='gray',
    
    max_words=75000,
    stopwords=stopwords
)

# generate the word cloud
DNC_wc.generate(DNC)

# Display the wordcloud
plt.imshow(DNC_wc, interpolation='bilinear')
plt.axis('off')
plt.show()
donkey = np.array(Image.open('../input/whitebg/DNCwhitebg.png'))
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

wc = WordCloud(background_color="#0b4980", width=1200, height=800, max_words=55000, mask=donkey, stopwords=stopwords, margin=1,random_state=2).generate(DNC)
# store default colored image
default_colors = wc.to_array()

#Custom Image
plt.figure(figsize=(50,25))
plt.title("DNC Speeches")

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")

#Text Processing
DNC = DNC.replace(".", ". ")
DNC = DNC.replace("announcer", ".")


plt.axis("off")
plt.figure(figsize=(50,25))
plt.title("Default colors")
plt.imshow(default_colors, interpolation="bilinear")
plt.axis("off")
plt.show()