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
with open ("/kaggle/input/data-for-text-analytics/Technical Skills for Job.txt") as fh:

    filedata=fh.read()
print("Sample Data:\n", filedata[:1000])
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
#Generate WordCloud data

wordcloud = WordCloud(stopwords=stopwords,max_words= 25,background_color="white").generate(filedata)

import matplotlib.pyplot as plt

plt.imshow(wordcloud)

plt.axis("off")

plt.show()

#Add redundant words as stopwwords 

stopwords.update(["experience","partner","solving","create","new","problem","use","Develop","working","Good","including","insight","insights","knowledge","technique","techniques","will","Strong","tool","etc","skills","Ability","solution","client","understanding","work","team","years","tools","teams","using","required"])
#Generate WordCloud data

wordcloud = WordCloud(width=800, height=800,stopwords=stopwords,max_words= 15,background_color="azure",min_font_size=10).generate(filedata)

import matplotlib.pyplot as plt

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud)

plt.axis("off")

plt.show()