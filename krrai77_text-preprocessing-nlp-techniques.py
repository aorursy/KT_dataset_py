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
#Load necessary libraries



import nltk

from nltk import FreqDist

import spacy

import matplotlib.pyplot as plt

import seaborn as sns
# View the data set



wines=pd.read_csv("/kaggle/input/wine-reviews/wine reviews.csv")

wines.head()
wines.drop("Sl.No.",axis=1,inplace=True)
# function to plot most frequent terms



def freq_words(x, terms = 30):

  all_words = ' '.join([text for text in x])

  all_words = all_words.split()



  fdist = FreqDist(all_words)

  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})



  # selecting top 20 most frequent words

  d = words_df.nlargest(columns="count", n = terms) 

  plt.figure(figsize=(20,5))

  ax = sns.barplot(data=d, x= "word", y = "count")

  ax.set(ylabel = 'Count')

  plt.show()
wines['Reviews Text'].fillna("Good",inplace=True)

wines['Reviews Title'].fillna("Neutral",inplace=True)
freq_words(wines['Reviews Text'])
freq_words(wines['Reviews Title'])
# remove unwanted characters, numbers and symbols

wines['Reviews Text'] = wines['Reviews Text'].str.replace("[^a-zA-Z#]", " ")

wines['Reviews Title'] = wines['Reviews Title'].str.replace("[^a-zA-Z#]", " ")
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
# function to remove stopwords

def remove_stopwords(rev):

    rev_new = " ".join([i for i in rev if i not in stop_words])

    return rev_new
# remove short words (length < 3)

wines['Reviews Text'] = wines['Reviews Text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

wines['Reviews Title'] = wines['Reviews Title'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))



# remove stopwords from the text

reviewstext = [remove_stopwords(r.split()) for r in wines['Reviews Text']]

reviewstitle = [remove_stopwords(r.split()) for r in wines['Reviews Title']]



# make entire text lowercase

reviewstext = [r.lower() for r in reviewstext]

reviewstitle = [r.lower() for r in reviewstitle]
freq_words(reviewstext, 35)
freq_words(reviewstitle, 35)
nlp = spacy.load('en', disable=['parser', 'ner'])



# filter noun and adjective

def lemmatization(texts, tags=['NOUN', 'ADJ']): 

       output = []

       for sent in texts:

             doc = nlp(" ".join(sent)) 

             output.append([token.lemma_ for token in doc if token.pos_ in tags])

       return output
tokenized_reviewstext = pd.Series(reviewstext).apply(lambda x: x.split())

print(tokenized_reviewstext[4])
tokenized_reviewstitle = pd.Series(reviewstitle).apply(lambda x: x.split())

print(tokenized_reviewstitle[6])
reviewstextlem = lemmatization(tokenized_reviewstext)

print(reviewstextlem[5])
reviewstitlelem = lemmatization(tokenized_reviewstitle)

print(reviewstitlelem[10])
reviewslemtext = []

for i in range(len(reviewstextlem)):

    reviewslemtext.append(' '.join(reviewstextlem[i]))



wines['reviewstext'] = reviewslemtext



freq_words(wines['reviewstext'], 35)
reviewslemtitle = []

for i in range(len(reviewstitlelem)):

    reviewslemtitle.append(' '.join(reviewstitlelem[i]))



wines['reviewstitle'] = reviewslemtitle



freq_words(wines['reviewstitle'], 35)
import PIL

from PIL import Image

from wordcloud import WordCloud, ImageColorGenerator
#Use a image for masking

wine_mask = np.array(Image.open("/kaggle/input/wineimage/wineimage.jpg"))

text = " ".join(review for review in wines.reviewstext)


# Create a word cloud image using a mask

wc = WordCloud(background_color="white", max_words=1000, mask=wine_mask)

               



# Generate a wordcloud

wc.generate(text)



# store to file

wc.to_file("/kaggle/working/winereviews.jpg")



# display the image

plt.figure(figsize=[20,10])

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
texttitle = " ".join(review for review in wines.reviewstitle)
# Generate a word cloud image without masking

wordcloud = WordCloud(background_color="black").generate(texttitle)





plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()