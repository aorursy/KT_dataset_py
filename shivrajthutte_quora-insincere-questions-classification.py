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
train=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")

train.head()
train.count()
test=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")

test.head()
test.count()
from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 
text1 = " ".join(review for review in train.question_text)

wordcloud = WordCloud(background_color="white").generate(text1)
plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
Negative_response = " ".join(review for review in train[train["target"]==0].question_text)

wordcloud0 = WordCloud(background_color="white", 

                          mode="RGBA", max_words=1000,).generate(Negative_response)
plt.imshow(wordcloud0, interpolation='bilinear')

plt.axis("off")

plt.show()
Positive_response = " ".join(review for review in train[train["target"]==1].question_text)

wordcloud1 = WordCloud(background_color="red", 

                          mode="RGBA", max_words=1000,).generate(Positive_response)
plt.imshow(wordcloud1, interpolation='bilinear')

plt.axis("off")

plt.show()
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.text import FreqDistVisualizer
vectorizer = CountVectorizer(stop_words='english')

docs       = vectorizer.fit_transform(text for text in train['question_text'])

features   = vectorizer.get_feature_names()



visualizer = FreqDistVisualizer(

    features=features#, size=(1080, 720)

)

visualizer.fit(docs)

visualizer.show()