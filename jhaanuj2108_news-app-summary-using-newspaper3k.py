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
# !pip install nltk
# !pip install newspaper3k
# !pip install PIL
# !pip install matplotlib
import nltk
from newspaper import Article
#Get the article
url = 'https://timesofindia.indiatimes.com/india/feluda-first-desi-covid-detector-gets-regulator-nod/articleshow/78224868.cms'
article = Article(url)
# Do some NLP
article.download() #Downloads the link’s HTML content
article.parse() #Parse the article
nltk.download('punkt')#1 time download of the sentence tokenizer
article.nlp()#  Keyword extraction wrapper
#Get the authors
article.authors
#Get the publish date 
article.publish_date
#Get the top image 
image_url = article.top_image

from PIL import Image
import requests
import matplotlib.pyplot as plt

response = requests.get(image_url, stream=True)
img = Image.open(response.raw)

plt.imshow(img)
plt.show()
#Get the article text
print(article.text)
#Get a summary of the article
print(article.summary)