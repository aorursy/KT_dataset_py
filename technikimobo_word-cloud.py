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
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
text=open('/kaggle/input/tweeting/tweet.txt','r').read()

words=text.split()
print('Number of words in text file :', len(words))
stopwords = ["wa"] + list(STOPWORDS)
stopwords=set(STOPWORDS)
tweets=WordCloud(background_color ='white',max_words=194963,stopwords=stopwords,width=3000, height=1800, min_font_size=50)
tweets.generate(text)

plt.imshow(tweets,interpolation='bilinear')
plt.figure(figsize = (30,30), facecolor = None)
plt.axis('off')
plt.show()