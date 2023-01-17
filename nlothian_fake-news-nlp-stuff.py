# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')



all_news = pd.read_csv("../input/fake.csv")

all_news.head(5)
print("Types and counts of stories", all_news.groupby(["type"]).size())

all_news.groupby(['type']).size().plot(kind='barh')
fake_news = all_news[all_news["type"] == "fake"]
import nltk



# Fill any blank fields

fake_news.title.fillna("", inplace=True)

fake_news.text.fillna("", inplace=True)



# Join the title and text

all_text = fake_news.title.str.cat(fake_news.text, sep=' ')



# Tokenize. The NLTK tokenizer isn't awesome. Spacy has a nice one, but I don't think it is installed

words = nltk.word_tokenize(" ".join(all_text.tolist()))
fake_news.head(20)
from nltk.corpus import stopwords

import string



# clearly more cleaning is needed here, but really I should get a better tokenizer

stop = stopwords.words('english') 

cleanwords = [i for i in words if i not in stop and i.isalpha() and len(i) > 2]
from wordcloud import WordCloud, STOPWORDS



wordcloud2 = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1200,

                          height=1000

                         ).generate(" ".join(cleanwords))





plt.imshow(wordcloud2)

plt.axis('off')

plt.show()
# Bigrams should be more interesting



bigrams = nltk.bigrams(cleanwords)
# look at the most common. 



from collections import Counter



counter = Counter(bigrams)

print(counter.most_common(10))
num_to_show = 30



labels = [" ".join(e[0]) for e in counter.most_common(num_to_show)]

values = [e[1] for e in counter.most_common(num_to_show)]



indexes = np.arange(len(labels))

width = 1



#plt.bar(indexes, values, width)

#plt.xticks(indexes + width * 0.5, labels, rotation=90)



plt.barh(indexes, values, width)

plt.yticks(indexes + width * 0.2, labels)

plt.show()