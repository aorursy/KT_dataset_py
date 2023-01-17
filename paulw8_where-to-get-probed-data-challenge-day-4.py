# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Viz

import matplotlib as mpl

import matplotlib.pyplot as plt



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from scipy.stats import ttest_ind

from PIL import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Import data. Using the scrubbed version to save on cleaning.

df = pd.read_csv("../input/scrubbed.csv")
# Let's take a peak at the data:

df.head(10)
# Let's create a graph so we know where to plan our travel.

sns.set(style="darkgrid")

sns.countplot(x="country", data=df).set_title("Best countries to get probed.")
# It would seem that "us" is the best location by a long way...
# Just to be sure before we book our plane tickets let's try a word map:



stopwords = set(STOPWORDS)

text = ' '.join(df.country[df['country'].isin(['us','au','gb','de','ca'])].astype(str))

    

wordcloud = WordCloud(

    background_color = 'white',

    stopwords = stopwords,

    max_words=25,

    max_font_size=100,

    random_state=42,

    width = 150,

    height = 150,

    collocations = False

).generate(text)



print(wordcloud)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()