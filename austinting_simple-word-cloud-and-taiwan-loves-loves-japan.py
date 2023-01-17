# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Learn from https://www.kaggle.com/sonuk7/simple-word-cloud-in-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/Shinzo Abe Tweet 20171024 - Tweet.csv')



# Any results you write to the current directory are saved as output.
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



ABE_WORD =""

for x in range(0, len(df['English Translation'])):

    stop = df['English Translation'][x].find('pic.twitter')

    ABE_WORD = ABE_WORD + df['English Translation'][x][:stop]+' Taiwan loves Japan Japan'

    



        



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white', 

                      max_words=300

                         ).generate(ABE_WORD)

plt.figure(figsize=(30,10))

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
