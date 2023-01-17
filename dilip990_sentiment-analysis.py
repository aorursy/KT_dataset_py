# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sample=pd.read_csv('../input/sample_submission.csv')
comb=train.append(test,ignore_index=True)
comb.shape
train['length']=train['tweet'].apply(len)
train.head()
train.describe()
train[train['length']==274]['tweet'].iloc[0]
train['length'].plot(bins=50,kind='hist')
train['tweet'].iloc[0]
train.head()
import re
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def remove_pattern(input_txt,pattern):
    r=re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
comb['tidy_tweet'] = np.vectorize(remove_pattern)(comb['tweet'],"@[\w]*")
comb.head()
comb['tidy_tweet']=comb['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
comb.head()
comb['tidy_tweet'] = comb['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
comb.head()
#bow_transformer = CountVectorizer(analyzer=text_process).fit(train['tweet'])
tokenized_tweet = comb['tidy_tweet'].apply(lambda x:x.split())
tokenized_tweet.head()
from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x])
tokenized_tweet.head()
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
comb['tidy_tweet'] = tokenized_tweet
comb.head()
all_words = ' '.join([text for text in comb['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation ="bilinear")
plt.axis('off')
plt.show()
normal_words = ' '.join([text for text in comb['tidy_tweet'][comb['label'] ==0]])
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()
racist_words = ' '.join([text for text in comb['tidy_tweet'][comb['label']==1]])
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size = 110).generate(racist_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()
def hashtag_extract(x):
    hashtag = []
    for i in x:
        ht = re.findall(r"#(\w+)",i)
        hashtag.append(ht)
        
    return hashtag
ht_regular = hashtag_extract(comb['tidy_tweet'][comb['label']==0])
ht_negative = hashtag_extract(comb['tidy_tweet'][comb['label']==1])
ht_regular = sum(ht_regular,[])
ht_negative = sum(ht_negative,[])
#print(ht_regular)
a = nltk.FreqDist(ht_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
a = nltk.FreqDist(ht_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(comb['tidy_tweet'])
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(comb['tidy_tweet'])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBRegressor

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]


xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
#lreg.fit(xtrain_bow, ytrain) 

#prediction = lreg.predict_proba(xvalid_bow) 
#prediction_int = prediction[:,1] >= 0.3  
#prediction_int = prediction_int.astype(np.int)

#f1_score(yvalid, prediction_int)
train_tfidf=tfidf[:31962,:]
test_tfidf=tfidf[31962:,:]
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
lreg.fit(xtrain_tfidf,ytrain)
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >=0.3
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid,prediction_int)
test_pred= lreg.predict_proba(test_tfidf)
test_pred_int=test_pred[:,1] >=0.3
test_pred_int = test_pred_int.astype(np.int)
test['label']=test_pred_int
submission = test[['id','label']]
submission.to_csv('123.csv',index=False)
