import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import re

import seaborn as sns

from IPython.display import display

pd.options.mode.chained_assignment = None

import matplotlib

matplotlib.style.use('ggplot')
tweets = pd.read_csv('../input/demonetization-tweets.csv', encoding = "ISO-8859-1")
tweets.head(5)
tweets.columns
a = tweets['statusSource'].str.split('/', expand=True)[4].str.split('"')

#display(tweets['statusSource'].str.split('/download/', expand=True))
source_ios = tweets['statusSource'].str.contains('iphone', na=False)

source_android = tweets['statusSource'].str.contains('android', na=False)
len(a)
import re
len(tweets['text'][source_android])
and_peops = tweets['text'][source_android]

and_peops = b.reset_index(drop=True)

ios_peops = tweets['text'][source_ios]

ios_peops = ios_peops.reset_index(drop=True)
tweets['text_new_and'] = ''

for i in range(len(and_peops)):

    m = re.search('(?<=:)(.*)', and_peops[i])

    try:

        tweets['text_new_and'][i] = m.group(0)

    except AttributeError:

        tweets['text_new_and'][i] = and_peops[i]
tweets['text_new_ios'] = ''

for i in range(len(ios_peops)):

    m = re.search('(?<=:)(.*)', ios_peops[i])

    try:

        tweets['text_new_ios'][i] = m.group(0)

    except AttributeError:

        tweets['text_new_ios'][i] = ios_peops[i]
tweets['tweetos_and'] = ''

for i in range(len(and_peops)):

    try:

        tweets['tweetos_and'][i] = and_peops.str.split(':')[i][0]

    except AttributeError:    

        tweets['tweetos_and'][i] = 'other'
tweets['tweetos_ios'] = ''

for i in range(len(and_peops)):

    try:

        tweets['tweetos_ios'][i] = and_peops.str.split(':')[i][0]

    except AttributeError:    

        tweets['tweetos_ios'][i] = 'other'
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



def wordcloud_by_province(tweets):

    stopwords = set(STOPWORDS)

    stopwords.add("https")

    stopwords.add("00A0")

    stopwords.add("00BD")

    stopwords.add("00A2")

    stopwords.add("00B8")

    stopwords.add("00AD")

    stopwords.add("co")

    stopwords.add("ed")

    stopwords.add("AMP")

    stopwords.add("demonetization")

    stopwords.add("Demonetization co")

    stopwords.add("lakh")

    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets['text_new_and'].str.upper()]))

    plt.imshow(wordcloud)

    plt.axis("off")

    #plt.title("Demonetization")



wordcloud_by_province(tweets)