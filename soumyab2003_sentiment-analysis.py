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
from collections import Counter
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
Apr_19=pd.read_csv('../input/review/reviews_reviews_app.zophop_201904.csv', encoding = 'UTF-16LE' )
May_19=pd.read_csv("../input/review/reviews_reviews_app.zophop_201905.csv", encoding = 'UTF-16LE')
June_19=pd.read_csv("../input/review/reviews_reviews_app.zophop_201906.csv", encoding = 'UTF-16LE')
july_19=pd.read_csv("../input/review/reviews_reviews_app.zophop_201907.csv", encoding = 'UTF-16LE')
Aug_19=pd.read_csv("../input/review/reviews_reviews_app.zophop_201908.csv", encoding = 'UTF-16LE')
Sept_19=pd.read_csv("../input/review/reviews_reviews_app.zophop_201909.csv", encoding = 'UTF-16LE')
Oct_19=pd.read_csv("../input/review/reviews_reviews_app.zophop_201910.csv", encoding = 'UTF-16LE')
Nov_19=pd.read_csv("../input/review/reviews_reviews_app.zophop_201911.csv", encoding = 'UTF-16LE')
Dec_19=pd.read_csv("../input/review/reviews_reviews_app.zophop_201912.csv", encoding = 'UTF-16LE')


Jan_20=pd.read_csv("../input/review/reviews_reviews_app.zophop_202001.csv")
feb_20=pd.read_csv("../input/review/reviews_reviews_app.zophop_202002.csv")
Mar_20=pd.read_csv("../input/review/reviews_reviews_app.zophop_202003.csv")
Jan_20.head()
Jan_20.shape
review_df=pd.concat([Apr_19,May_19,June_19,july_19,Aug_19,Sept_19,Oct_19,Nov_19,Dec_19,Jan_20,feb_20,Mar_20],axis=0,sort=False)
review_df.shape
review_df.head()
review_df.tail()
review_df['Review Text'].isnull().sum()
review_nonull=review_df.dropna(subset=['Review Text']).reset_index(drop=True)
review_nonull.shape
review_nonull.head()
review_nonull['Review Text'].apply(lambda cell: set([c.strip() for c in cell.split(' ')]))
proptype_uniq = review_nonull['Review Text'].apply(lambda cell: set(cell.split(' ')))
proptype_uniq
review_nonull['Review Text Cleaned'] = proptype_uniq
type(review_nonull['Review Text Cleaned'])
review_nonull.shape
review_nonull.head()
review_nonull['Review Text List'] = review_nonull['Review Text Cleaned'].apply(list)
review_nonull.head()
review_nonull['Review Text List'] = review_nonull['Review Text List'].apply(', '.join)
review_nonull['Review Text List']

review_nonull.head()
review_nonull.shape
review_nonull['Review Text List']

from nltk.probability import FreqDist
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append("really")
stop_words.append("please")
stop_words.append("thank")
stop_words.append("thanks")
stop_words.append("thank you")
stop_words.append("thank u")
stop_words.append("thanku")
stop_words.append("very")
stop_words.append("very very")
stop_words.append("very very very" )
stop_words.append("Very" )
stop_words.append("good" )
stop_words.append("nice" )
stop_words
# function to plot most frequent terms
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()),'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()
review_nonull['Review Text List'] = review_nonull['Review Text List'].str.replace("[^a-zA-Z#]", " ")
review_nonull['Review Text List']
review_nonull.shape
# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

# remove short words (length < 3)
review_nonull['Review Text List'] = review_nonull['Review Text List'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>4]))

# remove stopwords from the text
reviews = [remove_stopwords(r.split()) for r in review_nonull['Review Text List'] ]

# make entire text lowercase
reviews = [r.lower() for r in reviews]
freq_words(reviews, 10)
review_df['Star Rating'].isnull().sum()
Sentiment=[]
for i in review_df['Star Rating']:
    if i >=1 and i < 3:
        Sentiment.append("Negative")
    elif i == 3:
        Sentiment.append("Neutral")
    else:
        Sentiment.append("Positive")
    
    
    
review_df['Sentiment']=Sentiment

    
    
review_df.head()
review_df['Reviewer Language'].value_counts()
import seaborn as sns
sns.set(style="darkgrid")
ax = sns.countplot(x="Star Rating", data=review_df)

sns.set(style="darkgrid")
ax = sns.countplot(x="Sentiment", data=review_df)
review_df['Review Text'].isnull().sum()
no_text_rating=review_df.groupby('Star Rating')['Review Text'].apply(lambda x: x.isnull().sum())
no_text_rating
no_text_rating.plot(x="Star",y="Rating",kind = "bar")
review_df.head()
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
review_nonull['Star Rating']
Top_Rated1=review_df.loc[review_nonull['Star Rating'] >=4 ]
Top_Rated1.isnull().sum()

Top_Rated1=Top_Rated1.dropna(subset=['Review Text'])
Top_Rated1
Top_Rated=Top_Rated1
Top_Rated
Top_Rated.shape
text=Top_Rated['Review Text']

k = (' '.join(text))

wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud)
plt.axis('off');

stop_words.append("really")
stop_words.append("please")
stop_words.append("thank")
stop_words.append("thanks")
stop_words.append("thank you")
stop_words.append("thank u")
stop_words.append("thanku")
stop_words.append("very")
stop_words.append("very very")
stop_words.append("very very very" )
stop_words.append("amazing" )
stop_words.append("chalo" )
stop_words.append("Chalo" )
stop_words.append("thank" )
stop_words.append("useful")
stop_words.append("usefull")
stop_words.append("helpful")
stop_words.append("helpfull")
stop_words.append('excellent')

stop_words
Top_Rated.head()
Top_Rated = Top_Rated['Review Text'].str.replace("[^a-zA-Z#]", " ")
Top_Rated.shape
Top_Rated
Top_Rated.apply(lambda x: set([x.strip() for x in x.split(' ')]))
proptype_uniq = Top_Rated.apply(lambda x: set(x.split(' ')))
#proptype_uniq
Top_Rated = proptype_uniq
Top_Rated=Top_Rated.apply(list)

Top_Rated=Top_Rated.apply(', '.join)
Top_Rated
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
porter = PorterStemmer()
lancaster=LancasterStemmer()
stop_words.extend(['great','public','problem','number','avail','option','well','perfect','money','search','good','bad','very'])

Top_Rated = Top_Rated.apply(lambda x: ' '.join([w for w in x.split() if len(w)>4]))

Top_Rated
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
import string   
import re

def clean_text(text):
    ps=PorterStemmer()
    
    text=deEmojify(text) # remove emojis
    text_cleaned="".join([x for x in text if x not in string.punctuation]) # remove punctuation
    
    text_cleaned=re.sub(' +', ' ', text_cleaned) # remove extra white spaces
    text_cleaned=text_cleaned.lower() # converting to lowercase
    tokens=text_cleaned.split(" ")
    tokens=[token for token in tokens if token not in stop_words] # Taking only those words which are not stopwords
    text_cleaned=" ".join([ps.stem(token) for token in tokens])
    
    
    return text_cleaned
Top_Rated=Top_Rated.apply(lambda x:clean_text(x))
Top_Rated
Top_Rated = [remove_stopwords(r.split()) for r in Top_Rated ] 

Top_Rated = [r.lower() for r in Top_Rated]
Top_Rated
!python -m spacy download en # one time run
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output

tokenized_reviews = pd.Series(Top_Rated).apply(lambda x: x.split())
print(tokenized_reviews)
reviews_2 = lemmatization(tokenized_reviews)
print(reviews_2[2]) # print lemmatized review

reviews_3 = [] 
for i in range(len(reviews_2)): 
    reviews_3.append(' '.join(reviews_2[i]))

Top_Rated = reviews_3
print (len(Top_Rated))
while('' in Top_Rated) : 
    Top_Rated.remove('')
Top_Rated
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
    
    # function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

stop_words.extend(["show","work","city","wrong","update","showing","many","easy","need","working","proper","improvement","bus","month","waste","destination","useless","student","system","people","inform","love","right","help","use","updat","buse","today","person","download","peopl","real","poor","much","full","day","incorrect","user","sometim","applic","team","last","book","check","wonder","respon","late","wait","slow","destin","properli","current","instal","solv","wast","place","actual","open","life","buss","bhopal","transport","support","total","awesom","make"])
Top_Rated = [remove_stopwords(r.split()) for r in Top_Rated ] 
Len_genuine_review=len(Top_Rated)
Len_genuine_review
Top_Rated = [w.replace('time track', 'track') for w in Top_Rated]
Top_Rated = [w.replace('track time', 'track') for w in Top_Rated]
Top_Rated = [w.replace('locat track', 'track') for w in Top_Rated]
Top_Rated = [w.replace('track locat', 'track') for w in Top_Rated]
Top_Rated = [w.replace('time ticket', 'ticket') for w in Top_Rated]
Top_Rated = [w.replace('locat travel', 'travel') for w in Top_Rated]
Top_Rated = [w.replace('time conductor', 'conductor') for w in Top_Rated]
Top_Rated = [w.replace('exact track', 'track') for w in Top_Rated]
Top_Rated = [w.replace('exact conductor', 'conductor') for w in Top_Rated]
Top_Rated = [w.replace('actual track', 'track') for w in Top_Rated]
Top_Rated = [w.replace('track actual', 'track') for w in Top_Rated]
Top_Rated = [w.replace('track live', 'track') for w in Top_Rated]
Top_Rated = [w.replace('live track', 'track') for w in Top_Rated]
Top_Rated = [w.replace('livetrack', 'track') for w in Top_Rated]

Top_Rated = [w.replace('live time', 'track') for w in Top_Rated]
Top_Rated = [w.replace('live', 'track') for w in Top_Rated]
Top_Rated = [w.replace('correct track', 'track') for w in Top_Rated]
Top_Rated = [w.replace('track correct', 'track') for w in Top_Rated]
Top_Rated = [w.replace('exact', 'track') for w in Top_Rated]
#Top_Rated = [w.replace('pass', 'ticket') for w in Top_Rated]
Top_Rated = [w.replace('track travel', 'track') for w in Top_Rated]
Top_Rated = [w.replace('travel track', 'track') for w in Top_Rated]
Top_Rated = [w.replace('track book', 'ticket') for w in Top_Rated]
Top_Rated = [w.replace('actual track', 'track') for w in Top_Rated]
Top_Rated = [w.replace('actual locat', 'track') for w in Top_Rated]
Top_Rated = [w.replace('time', 'track') for w in Top_Rated]
Top_Rated = [w.replace('locat', 'track') for w in Top_Rated]
Top_Rated = [w.replace('rout', 'track') for w in Top_Rated]
Top_Rated = [w.replace('correct', 'track') for w in Top_Rated]
Top_Rated = [w.replace('travel', 'track') for w in Top_Rated]
#Top_Rated = [w.replace('card', 'ticket') for w in Top_Rated]


freq_words(Top_Rated,5)

Top_Rated = ' '.join([text for text in Top_Rated])
Top_Rated = Top_Rated.split()
fdist = FreqDist(Top_Rated)
words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
words_df= words_df.nlargest(columns="count", n = 5)
    
    

words_df=words_df.reset_index(drop=True)
words_df['percentage']=(words_df['count']/len(Top_Rated1)*100)
words_df
words_df[' Genuine percentage']=(words_df['count']/Len_genuine_review*100)
words_df.drop(columns=['percentage'])
def getMostCommon(reviews_list,topn=5):
    reviews=" ".join(reviews_list)
    tokenised_reviews=reviews.split(" ")
    
    
    freq_counter=Counter(tokenised_reviews)
    return freq_counter.most_common(topn)
def plotMostCommonWords(reviews_list,topn=5,title="Common Review Words",color="blue",axis=None): #default number of words is given as 20
    top_words=getMostCommon(reviews_list,topn=topn)
    data=pd.DataFrame()
    data['words']=[val[0] for val in top_words]
    data['freq']=[val[1] for val in top_words]
    if axis!=None:
        sns.barplot(y='words',x='freq',data=data,color=color,ax=axis).set_title(title+" top "+str(topn))
    else:
        sns.barplot(y='words',x='freq',data=data,color=color).set_title(title+" top "+str(topn))
rcParams['figure.figsize'] = 5,5
fig,ax=plt.subplots(1,2)
fig.subplots_adjust(wspace=1)
plotMostCommonWords(Top_Rated,5,"Top_Rated",axis=ax[0])
least_Rated1=review_df.loc[review_df['Star Rating'] <=2]
least_Rated1.isnull().sum()

least_Rated1=least_Rated1.dropna(subset=['Review Text'])
least_Rated1.head()
least_Rated=least_Rated1
len(least_Rated)
least_Rated = least_Rated['Review Text'].str.replace("[^a-zA-Z#]", " ")
least_Rated
least_Rated.apply(lambda x: set([x.strip() for x in x.split(' ')]))
proptype_uniq_least = least_Rated.apply(lambda x: set(x.split(' ')))

least_Rated = proptype_uniq_least
least_Rated=least_Rated.apply(list)

least_Rated=least_Rated.apply(', '.join)
least_Rated
least_Rated = least_Rated.apply(lambda x: ' '.join([w for w in x.split() if len(w)>4]))
least_Rated=least_Rated.apply(lambda x:clean_text(x))
least_Rated  = [remove_stopwords(r.split()) for r in least_Rated ] 
least_Rated = [r.lower() for r in least_Rated]
least_Rated
tokenized_reviews_least = pd.Series(least_Rated).apply(lambda x: x.split())
print(tokenized_reviews_least)
reviews_least_2=lemmatization(tokenized_reviews_least)
print(reviews_least_2[10])
reviews_3_least = [] 
for i in range(len(reviews_least_2)): 
    reviews_3_least.append(' '.join(reviews_least_2[i]))

least_Rated = reviews_3_least
least_Rated
while('' in least_Rated) : 
    least_Rated.remove('')
least_Rated
stop_words.extend(['fake','hour'])
least_Rated = [remove_stopwords(r.split()) for r in least_Rated ] 
least_Rated_length=len(least_Rated)
least_Rated_length
least_Rated = [w.replace('time track', 'track') for w in least_Rated]
least_Rated = [w.replace('track time', 'track') for w in least_Rated]
least_Rated = [w.replace('locat track', 'track') for w in least_Rated]
least_Rated = [w.replace('track locat', 'track') for w in least_Rated]
least_Rated = [w.replace('time ticket', 'ticket') for w in least_Rated]
least_Rated = [w.replace('locat travel', 'travel') for w in least_Rated]
least_Rated = [w.replace('time conductor', 'conductor') for w in least_Rated]
least_Rated = [w.replace('exact track', 'track') for w in least_Rated]
least_Rated = [w.replace('exact conductor', 'conductor') for w in least_Rated]
least_Rated = [w.replace('actual track', 'track') for w in least_Rated]
least_Rated = [w.replace('track actual', 'track') for w in least_Rated]
least_Rated = [w.replace('track live', 'track') for w in least_Rated]
least_Rated = [w.replace('live track', 'track') for w in least_Rated]
least_Rated = [w.replace('livetrack', 'track') for w in least_Rated]

least_Rated = [w.replace('live time', 'track') for w in least_Rated]
least_Rated = [w.replace('live', 'track') for w in least_Rated]
least_Rated = [w.replace('correct track', 'track') for w in least_Rated]
least_Rated = [w.replace('track correct', 'track') for w in least_Rated]
least_Rated = [w.replace('exact', 'track') for w in least_Rated]
#Top_Rated = [w.replace('pass', 'ticket') for w in Top_Rated]
least_Rated = [w.replace('track travel', 'track') for w in least_Rated]
least_Rated = [w.replace('travel track', 'track') for w in least_Rated]
least_Rated = [w.replace('track book', 'ticket') for w in least_Rated]
least_Rated = [w.replace('actual track', 'track') for w in least_Rated]
least_Rated = [w.replace('actual locat', 'track') for w in least_Rated]
least_Rated= [w.replace('locat time', 'track') for w in least_Rated]
least_Rated= [w.replace('track correct', 'track') for w in least_Rated]
least_Rated= [w.replace('tracker locat ', 'track') for w in least_Rated]
least_Rated = [w.replace('track locat', 'track') for w in least_Rated]
least_Rated = [w.replace('track rout', 'track') for w in least_Rated]
least_Rated = [w.replace('rout track', 'track') for w in least_Rated]
least_Rated = [w.replace('pass rout', 'pass') for w in least_Rated]
least_Rated = [w.replace('travel track', 'track') for w in least_Rated]
least_Rated = [w.replace('track travel', 'track') for w in least_Rated]
least_Rated= [w.replace('time', 'track') for w in least_Rated]
least_Rated= [w.replace('correct', 'track') for w in least_Rated]
least_Rated= [w.replace('locat', 'track') for w in least_Rated]
least_Rated = [w.replace('rout', 'track') for w in least_Rated]
least_Rated = [w.replace('travel', 'track') for w in least_Rated]
freq_words(least_Rated ,5)
while('' in least_Rated) : 
    least_Rated.remove('')
rcParams['figure.figsize'] = 5,5
fig,ax=plt.subplots(1,2)
fig.subplots_adjust(wspace=1)
plotMostCommonWords(least_Rated,5,"least_Rated",axis=ax[0])
least_Rated = ' '.join([text for text in least_Rated])
least_Rated = least_Rated.split()
fdist = FreqDist(least_Rated)
least_Rated_words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
least_Rated_words_df= least_Rated_words_df.nlargest(columns="count", n = 5)
    
    
least_Rated_words_df=least_Rated_words_df.reset_index(drop=True)
least_Rated_words_df['Percentage']=(least_Rated_words_df['count']/len(least_Rated1)*100)
least_Rated_words_df
least_Rated_words_df['Genuine review Percentage']=(least_Rated_words_df['count']/least_Rated_length*100)
least_Rated_words_df.drop(columns=['Percentage'])

review_nonull.head()
All_reviews = review_nonull['Review Text List'].str.replace("[^a-zA-Z#]", " ")
All_reviews = All_reviews.apply(lambda x: ' '.join([w for w in x.split() if len(w)>4]))
All_reviews
All_reviews=All_reviews.apply(lambda x:clean_text(x))
All_reviews  = [remove_stopwords(r.split()) for r in All_reviews ] 
All_reviews = [r.lower() for r in All_reviews]
tokenized_reviews_all = pd.Series(All_reviews).apply(lambda x: x.split())
print(tokenized_reviews_all)
reviews_all=lemmatization(tokenized_reviews_all)
print(reviews_all[11])
reviews_3_all = [] 
for i in range(len(reviews_all)): 
    reviews_3_all.append(' '.join(reviews_all[i]))

All_Reviews= reviews_3_all
while('' in All_reviews) : 
    All_reviews.remove('')

All_reviews
All_reviews = [w.replace('locat track', 'track') for w in All_reviews]
All_reviews = [w.replace('track locat', 'track') for w in All_reviews]
All_reviews = [w.replace('locat accur', 'track') for w in All_reviews]
All_reviews = [w.replace('accur locat', 'track') for w in All_reviews]
All_reviews = [w.replace('locate accur', 'track') for w in All_reviews]
freq_words(All_reviews,10)
subs="locat"
res=[i for i in All_reviews if subs in i]
res
