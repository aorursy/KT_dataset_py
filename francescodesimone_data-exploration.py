# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

from nltk.corpus import stopwords



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output





topics_list = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
for ind, topic in enumerate(topics_list):

    tags = np.array(pd.read_csv("../input/"+topic+".csv", usecols=['tags'])['tags'])

    text = ''

    for ind, tag in enumerate(tags):

        text = " ".join([text, tag])

    text = text.strip()

    

    wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)

    wordcloud.recolor(random_state=ind*312)

    plt.imshow(wordcloud)

    plt.title("Wordcloud for topic : "+topic)

    plt.axis("off")

    plt.show()

dataframes = {

    "cooking": pd.read_csv("../input/cooking.csv"),

    "crypto": pd.read_csv("../input/crypto.csv"),

    "robotics": pd.read_csv("../input/robotics.csv"),

    "biology": pd.read_csv("../input/biology.csv"),

    "travel": pd.read_csv("../input/travel.csv"),

    "diy": pd.read_csv("../input/diy.csv"),

}

print(dataframes["robotics"].iloc[1])