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
import pandas as pd

import re

import nltk

import matplotlib.pyplot as plt

hd = pd.read_csv("../input/a-wealth-of-data-sunshine/hair_dryer01.csv")

mw = pd.read_csv("../input/a-wealth-of-data-sunshine/microwave01.csv")

pf = pd.read_csv("../input/a-wealth-of-data-sunshine/pacifier01.csv")
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('[[VIDEOID:]', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('\n', '', text)

    text = re.sub('<br/>', '', text)

    text = re.sub('&#34', '', text)

    text = re.sub('/', '', text)

    text = re.sub('鈥檚','',text)

    text = re.sub('鈥檛', '', text)

    text = re.sub('鈥檓' ,'', text)

    text = re.sub('馃憤馃徑', '', text)

  

    return text
mw['review_body'] = mw['review_body'].apply(lambda x: clean_text(x))

hd['review_body'] = hd['review_body'].apply(lambda x: clean_text(x))

pf['review_body']
mwr = mw[mw['star_rating']>=4]

mwrg = mwr["review_body"]

mwre = mw[mw['star_rating']<=2]

mwren = mwre["review_body"]
hdr  = hd[hd["star_rating"]>=4]

hdrg = hdr["review_body"]

hdre = hd[hd["star_rating"]<=2]

hdren = hdre["review_body"]
pfr  = pf[pf["star_rating"]>=4]

pfrg = pfr["review_body"]

pfre = pf[pf["star_rating"]<=2].astype(str)

pfren = pfre["review_body"]
coms = mw["Sentiment_polarity"]

coms1 = hd["Sentiment_polarity"]

coms2 = pf["Sentiment_polarity"]
data1 = mw[coms>=0.5]

data2 = mw[coms<0]
data3 = hd[coms1>=0.5]

data4 = hd[coms1<0]
data5 = pf[coms2>=0.5]

data6 = pf[coms2<0]
from wordcloud import WordCloud

fig, (ax1, ax2 ) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(mwrg))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Microwave_Good_Review',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(data1["review_body"]))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Microwave_Pos_Review',fontsize=40);
from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(mwren))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Microwave_Bad_Review',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(data2["review_body"]))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Microwave_Neg_Review',fontsize=40);
from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(hdrg))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Hair_dryer_Good_Review',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(data3["review_body"]))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Hair_dryer_Pos_Review',fontsize=40);
from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(hdren))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Hair_dryer_Bad_Review',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(data4["review_body"]))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Hair_dryer_Neg_Review',fontsize=40);
from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(pfren))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Pacifier_Good_Review',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(data5["review_body"]))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Pacifier_Pos_Review',fontsize=40);
import nltk

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

data1["review_body"] = data1["review_body"].apply(lambda x: tokenizer.tokenize(x))

data2["review_body"] = data2["review_body"].apply(lambda x: tokenizer.tokenize(x))
from nltk.corpus import stopwords

def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words





data1['review_body'] = data1['review_body'].apply(lambda x : remove_stopwords(x))

data2['review_body'] = data2['review_body'].apply(lambda x : remove_stopwords(x))

data2
pos = pd.DataFrame(data1["review_body"])

pos = np.array(pos)

pos.tolist()



neg = pd.DataFrame(data2["review_body"])

neg = np.array(neg)

neg = neg.tolist()
from gensim import corpora,models
neg_dict=corpora.Dictionary(neg[2])

neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]]

neg_corpus
neg_lda = models.LdaModel(neg_corpus,num_topics=5,id2word = neg_dict)

for i in range(5):

    print('topic',i)

    print(neg_lda.print_topic(i))
pos_dict=corpora.Dictionary(pos[2])

pos_corpus = [pos_dict.doc2bow(i) for i in pos[2]]

pos_corpus
# After preprocessing, the text format

def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



mw['review_body'] = mw['review_body'].apply(lambda x : combine_text(x))

mw['review_body']
# text preprocessing function

def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return combined_text
mw['review_body'] = mw['review_body'].apply(lambda x : text_preprocessing(x))

mw['review_body'].head()
from textblob import TextBlob



def find_pol(review):

    return TextBlob(review).sentiment.polarity



mw['Sentiment_Polarity'] = mw['review_body'].apply(find_pol)



mw[mw['star_rating']>4]
pf['review_body'] = pf['review_body'].astype(str)

pf['review_body']
from textblob import TextBlob

text_blob_object = TextBlob(pf['review_body'][350])

print(text_blob_object.sentiment)
def find_pol(review):

    return TextBlob(review).sentiment.polarity



pf['Sentiment_Polarity'] = pf['review_body'].apply(find_pol)



pf.head()
def find_pol(review):

    return TextBlob(review).sentiment.polarity



mw['Sentiment_Polarity'] = mw['review_body'].apply(find_pol)



mw.head()
TextBlob('disappoint').sentiment
import seaborn as sns

sns.distplot(mw['Sentiment_Polarity'])

mw["Sentiment_Polarity"].mean()
output = pd.DataFrame({'Sentiment_polarity':pf["Sentiment_Polarity"] })

output.to_csv('submission.csv', index=False)
import seaborn as sns

sns.distplot(pf['Sentiment_Polarity'])

pf["Sentiment_Polarity"].mean()
from textblob import TextBlob

text_blob_object = TextBlob("fast")

print(text_blob_object.sentiment)
def find_pol(review):

    return TextBlob(review).sentiment.polarity



hd['Sentiment_Polarity'] = hd['review_body'].apply(find_pol)



hd.head()
import seaborn as sns

sns.distplot(hd['Sentiment_Polarity'],rug=True, hist=False)

sns.distplot(pf['Sentiment_Polarity'],rug=True, hist=False)
sns.distplot(hd['Sentiment_Polarity'],rug=True, hist=True)
hd["Sentiment_Polarity"].mean()

# pf["Sentiment_Polarity"].mean()

# mw["Sentiment_Polarity"].mean()