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
import glob, json

root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

all_json
data=pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

print(data.shape)

data.head()
abstract=data['abstract'].dropna()

abstract
abstract=list(abstract)

abstract
publish=data['publish_time'].dropna()

publish
journals=data['journal'].dropna()

journals
#summary

data.describe()
# Nan values in all columns

data.isna().sum()
#Titles

title=data['title'].dropna()

title
!git clone https://github.com/amueller/word_cloud.git
import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt
#c:\intelpython3\lib\site-packages\matplotlib\__init__.py:

import warnings

warnings.filterwarnings("ignore")
# Create and generate a word cloud image:

wordcloud = WordCloud().generate(abstract[0])



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(abstract[0])

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
# Save the image in the img folder:

#wordcloud.to_file("img/first_review.png")
text = " ".join(review for review in abstract)
print ("There are {} words in the combination of all review.".format(len(text)))
# Create stopword list:

stopwords = set(STOPWORDS)

stopwords.update(["covid", "corona", "disease", "virus", "infection"])



# Generate a word cloud image

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)



# Display the generated image:

# the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()