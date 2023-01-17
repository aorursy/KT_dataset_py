# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

import re



cachedStopWords = stopwords.words("english")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))

filename = check_output(["ls", "../input"]).decode("utf8").rstrip("\n")

print(filename)



file = "../input/" + filename

csv_file = pd.read_csv(file)

list_tweet_en = []

list_tweet_en = csv_file['English Translation']



list_tweet_word = []

buffer = []

from nltk.tokenize import word_tokenize

for line in list_tweet_en:

    buffer = buffer + word_tokenize(line)

    

list_tweet_word = buffer

#print(list_tweet_word)

from collections import Counter



#data = ['aaa', 'bbb', 'ccc', 'aaa', 'ddd']



result = []

counter = Counter(list_tweet_word)

for word, cnt in counter.most_common():

    if re.match("\w+", word) and word not in cachedStopWords:

        result.append(word + "(" + str(cnt) + ")")



print(result)





######################

######################

text = ""

#print(list_tweet_en)

for item in list_tweet_en:

    text = text + " " + item

#somewords = "We the People of the United States, in Order to form a more perfect Union, establish Justice"



wordcloud = WordCloud().generate(text)



# Display the generated image:

# the matplotlib way:



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



# lower max_font_size

wordcloud = WordCloud(max_font_size=40).generate(text)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()



# Any results you write to the current directory are saved as output.