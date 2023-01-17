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
import warnings

warnings.filterwarnings("ignore") #To ignore warnings to get clean output

import pandas as pd

import matplotlib.pyplot as plt
data=pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")

print(len(data))#total number of entries

data.head()

data.isnull().sum()
data.drop(columns=['negativereason','negativereason_confidence',

                  'airline_sentiment_gold','tweet_coord','negativereason_gold','tweet_location','user_timezone'

                  ,'tweet_id','retweet_count','name','tweet_created','airline_sentiment_confidence'],inplace=True)
data['token_length'] = [len(x.split(" ")) for x in data.text]

max(data.token_length)
data.airline_sentiment.value_counts()
data['airline'].unique()
count_neg={}

count_pos={}

count_neu={}

for i in data['airline'].unique():

    x=len(data.loc[(data['airline']==i) & (data['airline_sentiment']=='negative')])

    count_neg.update({i:x})

for i in data['airline'].unique():

    x=len(data.loc[(data['airline']==i) & (data['airline_sentiment']=='positive')])

    count_pos.update({i:x})

for i in data['airline'].unique():

    x=len(data.loc[(data['airline']==i) & (data['airline_sentiment']=='neutral')])

    count_neu.update({i:x})
print(count_neg)

print(count_pos)

print(count_neu)
import numpy as np

# set width of bar

barWidth = 0.25

plt.figure(figsize=(20,10))

 

# set height of bar

bars1 = count_neg.values()

bars2 = count_pos.values()

bars3 = count_neu.values()

 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

 

# Make the plot

plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='neg')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='pos')

plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='neu')

 

# Add xticks on the middle of the group bars

plt.xlabel('Airlines', fontweight='bold',fontsize=18)

plt.xticks([r + barWidth for r in range(len(bars1))], [i for i in data['airline'].unique()],fontsize=16)

                                                       

 

# Create legend & Show graphic

plt.legend(fontsize="x-large")

plt.show()



from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer



tvec = TfidfVectorizer(stop_words=None, max_features=100000, ngram_range=(1, 3))

lr = LogisticRegression()



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score, recall_score, f1_score



def lr_cv(splits, X, Y, pipeline, average_method):

    

    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)

    accuracy = []

    precision = []

    recall = []

    f1 = []

    for train, test in kfold.split(X, Y):

        lr_fit = pipeline.fit(X[train], Y[train])

        prediction = lr_fit.predict(X[test])

        scores = lr_fit.score(X[test],Y[test])

        

        accuracy.append(scores * 100)

        precision.append(precision_score(Y[test], prediction, average=average_method)*100)

        print('              negative    neutral     positive')

        print('precision:',precision_score(Y[test], prediction, average=None))

        recall.append(recall_score(Y[test], prediction, average=average_method)*100)

        print('recall:   ',recall_score(Y[test], prediction, average=None))

        f1.append(f1_score(Y[test], prediction, average=average_method)*100)

        print('f1 score: ',f1_score(Y[test], prediction, average=None))

        print('-'*50)



    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))

    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))

    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))

    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
from nltk.tokenize import TweetTokenizer

import re

from nltk.corpus import stopwords

from nltk.stem import LancasterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

import nltk



def clean_tweet(tweet):

    return ''.join(re.sub(r"(@[A-Za-z0-9]+)|(http\S+)|(#[A-Za-z0-9]+)|(\$[A-Za-z0-9]+)|(RT)|([0-9]+)","",tweet))

def remove_special_chars(tweets):  # it unrolls the hashtags to normal words

    for remove in map(lambda r: re.compile(re.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",

                                                                     "@", "%", "^", "*", "(", ")", "{", "}",

                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",

                                                                     "!", "?", ".", "'",

                                                                     "--", "---", "#"]):

        tweets.replace(remove, "", inplace=True)

    return tweets

lem=WordNetLemmatizer()

tkn=TweetTokenizer()

ps=LancasterStemmer()

pd.options.display.max_colwidth=1000



def filter_tweet(tweet):

    filtered=[]

    for w in tweet:

        if w.lower() not in stopwords.words('english'):

            filtered.append(w)

    return filtered

def get_pos(word):

    tag=nltk.pos_tag([word])[0][1][0]

    if tag =='J':

        return wordnet.ADJ

    elif tag =='V':

        return wordnet.VERB

    elif tag =='N':

        return wordnet.NOUN

    elif tag =='R':

        return wordnet.ADV

    else:

        return wordnet.NOUN
data['cleantweet']=data['text'].apply(lambda row: clean_tweet(row))

remove_special_chars(data.cleantweet)

data.head()
data['tokenized_text'] = data.apply(lambda row : tkn.tokenize(row['cleantweet']), axis=1)





data['filteredsent'] = data['tokenized_text']#.apply(lambda row : filter_tweet(row))





data['Lemmatized']=data.apply(lambda row :[lem.lemmatize(i,pos=get_pos(i)) for i in row['filteredsent']],axis=1)





data['stemwords'] = data.apply(lambda row : [ps.stem(i) for i in row['filteredsent']],axis=1)





#The final sentence is made from lemetized words.It can be changed to stemmed words.

#Totally upto user.This sentence will be input to sklearn's feature extractor.

data['prtext']=data['Lemmatized'] 





data['prtext']=data['prtext'].apply(lambda row : ' '.join(row))

data.tail()
lr_cv(5, data.prtext, data.airline_sentiment, original_pipeline, 'macro')
from imblearn.pipeline import make_pipeline

from imblearn.over_sampling import SMOTE

SMOTE_pipeline = make_pipeline(tvec, SMOTE(random_state=777),lr)
lr_cv(5, data.text, data.airline_sentiment, SMOTE_pipeline, 'macro')
