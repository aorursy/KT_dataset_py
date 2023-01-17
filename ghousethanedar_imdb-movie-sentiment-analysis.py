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
#importing the training data

imdb_data=pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

print(imdb_data.shape)

imdb_data.head(10)
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
import nltk

import string

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords
## function



review_data_list=[]

indi_lines=imdb_data['review'].values.tolist()
for i in indi_lines:

    #creating the word tokenize and removing the puntuations

    rem_tokenizer=RegexpTokenizer(r'\w+')

    words_tokens=rem_tokenizer.tokenize(i)

    

    #converting the words to lower case

    

    low_words=[w.lower() for w in words_tokens]

    

    #invoke all the english stop words

    

    stop_words_list=set(stopwords.words('english'))

    

    #removing the stopwords

    rem_stop_words=[w for w in low_words if w not in stop_words_list]

    

    review_data_list.append(rem_stop_words)

            

    
print(len(review_data_list))
review_data_list[0:5]
import gensim

embedding_dim=100



# training the gensim model



model=gensim.models.Word2Vec(sentences=review_data_list,size=embedding_dim,workers=4,min_count=1)



words=list(model.wv.vocab)



print(len(words))
model.wv.most_similar('amazing')
model.wv.most_similar('awful')
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer(ngram_range=(1,1),stop_words='english')
vec.fit(words)

words_dtm=vec.transform(words)
words_dtm
imdb_data.head()
imdb_data['sentiment']=imdb_data['sentiment'].map({'positive':0,'negative':1})
x=imdb_data['review']

y=imdb_data['sentiment']
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=123)
stop_words_list=set(stopwords.words('english'))

    

#removing the stopwords

    

#xtrain=[w for w in list(xtrain.values) if w not in stop_words_list]



#xtest=[w for w in list(xtest.values) if w not in stop_words_list]
vec.fit(xtrain)

xtrain_dtm=vec.transform(xtrain)

xtest_dtm=vec.transform(xtest)
vec.vocabulary_
xtrain_dtm
xtest_dtm

from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()

mnb
ytrain.head()
mnb.fit(xtrain_dtm,ytrain)
ypred=mnb.predict(xtest_dtm)
from sklearn import metrics
print(metrics.classification_report(ytest,ypred))
print(metrics.confusion_matrix(ytest,ypred))
xtest[ytest>ypred]
xtest[5166]
xtrain_tokens=vec.get_feature_names()
print(xtrain_tokens[0:50])
print(xtrain_tokens[-50:])
mnb.feature_count_.shape
# number of times each token appears across all HAM messages

pos_token_count = mnb.feature_count_[0, :]

pos_token_count
# number of times each token appears across all SPAM messages

neg_token_count = mnb.feature_count_[1, :]

neg_token_count
# create a DataFrame of tokens with their separate ham and spam counts

tokens = pd.DataFrame({'token':xtrain_tokens, 'Pos':pos_token_count, 'Neg':neg_token_count}).set_index('token')

tokens.head()
# add 1 to ham and spam counts to avoid dividing by 0

tokens['Pos'] = tokens.Pos + 1

tokens['Neg'] = tokens.Neg + 1

tokens.sample(5, random_state=6)
# examine 5 random DataFrame rows

tokens.sample(5, random_state=6)
# convert the ham and spam counts into frequencies

tokens['Pos'] = tokens.Pos / mnb.class_count_[0]

tokens['Neg'] = tokens.Neg / mnb.class_count_[1]

tokens.sample(5, random_state=6)
# calculate the ratio of spam-to-ham for each token

tokens['Neg_ratio'] = tokens.Neg / tokens.Pos

tokens.sample(5, random_state=6)
# examine the DataFrame sorted by spam_ratio

tokens.sort_values('Neg_ratio', ascending=False)
import matplotlib.pyplot as plt # visualization

import seaborn as sns # visualization 

from wordcloud import WordCloud, STOPWORDS # this module is for making wordcloud in python
# difine wordcloud function from wordcloud library. set some parameteres for beatuful plotting

wc = WordCloud()

# generate word cloud using df_yelp_tip_top['text_clear']

wc.generate(str(imdb_data['review']))

# declare our figure 

plt.figure(figsize=(20,10), facecolor='k')

# add title to the graph

plt.title("Most frequent words in Imdb dataset", fontsize=40,color='white')

plt.imshow(wc)

plt.show()