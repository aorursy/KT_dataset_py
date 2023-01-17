# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import matplotlib.pyplot as plt
%matplotlib inline
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import nltk
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
print('train',train.shape)
print('test',train.shape)
train.head()
train['text'][1000]
print('train,null%\n',train.isnull().mean())
print()
print('test,null%\n',test.isnull().mean())
#In train,  keyword has 0.8% and loc has 33% missing values
#In test,  keyword has 0.7% and loc has 33% missing values
disaster=train[train['target']==1]
non_dist=train[train['target']==0]

d=disaster['keyword'].value_counts()[:20]
nd=non_dist['keyword'].value_counts()[:20]
from plotly.subplots import make_subplots
fig=make_subplots(rows=1,cols=2)
fig.add_traces(go.Bar(y=d.index,x=d.values,orientation='h',name='disaster keywords'),1,1)
fig.add_traces(go.Bar(y=nd.index,x=nd.values,orientation='h',name='non_dist kewords'),1,2)
loc_d=disaster['location'].value_counts()[:10]
loc_nd=non_dist['location'].value_counts()[:10]
fig=make_subplots(rows=1,cols=2)
fig.add_traces(go.Bar(y=loc_d.index,x=loc_d.values,orientation='h',name='Most disaster loc'),1,1)
fig.add_traces(go.Bar(y=loc_nd.index,x=loc_nd.values,orientation='h',name='non_dist Loc '),1,2)

#USA records more disaster teeets
#Newyork less disaster tweets
import plotly.figure_factory as ff
#calculate len of tweets in both disaster and non
d1=disaster['text'].apply(len)
nd1=non_dist['text'].apply(len)
ff.create_distplot([d1,nd1],['len_d1','len_nd1'])
print('mean_len_d1',d1.mean())
print('mean_len_nd1',nd1.mean())

#disaster tweets len is more comp to non diasster
non_dist['text'].str.len().sort_values(ascending=False)[:10]
#to remove url,punc,#,@,stopwords
print(non_dist['text'][1270])
print(non_dist['text'][4801])
print(non_dist['text'][261])
print(non_dist['text'][5379])

disaster['text'].str.len().sort_values(ascending=False)[:10]

print(disaster['text'][614])
print(disaster['text'][635])
print(disaster['text'][2718])
print(disaster['text'][1111])
import string
import re
'''
import re
a='http://t.co/FYJWjDkM5I this is it'
a=re.sub('http://[a-z]+\.[a-z]+/[a-zA-Z]+','',non_dist['text'][6555],)
#re.findall('http://[a-z]+\.[a-z]+/[a-zA-Z]+',a)

re.sub('[^\w]',' ',a)
'''

'''a='@IcyMagistrate ÛÓher upper armÛÒ those /friggin/ icicle projectilesÛÒ and leg from various other wounds the girl looks like a miniature moreÛÓ'
a.encode("ascii", errors="ignore").decode()
#output=='@IcyMagistrate her upper arm those /friggin/ icicle projectiles and leg from various other wounds the girl looks like a miniature'http.?://[a-z]+\.[a-z]+/[a-zA-Z0-9]+ more'
''''http.?://[a-z]+\.[a-z]+/[a-zA-Z0-9]+'''

#a= train['text'][614]
def clean_text(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text= emoji_pattern.sub(r'', text)
    text=text.encode('ascii',errors='ignore').decode()
    #print(text)
    text=text.lower()
    text=re.sub('http.?://[a-z]+\.[a-z]+/[a-zA-Z0-9]+', '',text)
    text=re.sub('&amp',' and',text)
    text=re.sub('gt','greater than',text)
    text=re.sub('lt','lesser than',text)
    text=re.sub('rt','retweet',text)
    text=re.sub('n\'t',' not',text)
    text=re.sub('\'s',' is',text)
    text=re.sub('\'ll', 'will',text)
    text=re.sub('\'ve','have',text)
    text=re.sub('i\'m','i am',text)
    text=re.sub('\'re',' are',text)
    #print(text)
    #print(text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    #print(text)
    #print(text)
    text = re.sub('\w*\d\w*','', text)
    return text
train['new_text']=train['text'].apply(lambda x: clean_text(x))
test['new_text']=train['text'].apply(lambda x: clean_text(x))
train['new_text'][1270]
#import worldcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
w=WordCloud().generate(train['new_text'][1])
plt.imshow(w, interpolation='bilinear')
plt.axis('off')
disaster_train=train[train['target']==1]
nondisaster=train[train['target']==0]
fig,ax=plt.subplots(1,2,figsize=[30,10])
w1=WordCloud(background_color='white').generate(''.join(disaster_train['new_text']))
ax[0].imshow(w1)
ax[0].axis('off')
ax[0].set_title('Disaster Tweets',fontsize=40);

w2=WordCloud(background_color='white').generate(''.join(nondisaster['new_text']))
ax[1].imshow(w2)
ax[1].axis('off')
ax[1].set_title('Non_Disaster Tweets',fontsize=40);
tokens=word_tokenize(train['text'][0])
print('normal tokens',tokens)
tokens=[t for t in tokens if t not in stopwords.words('english')]
print('no stipwords',tokens)
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
snowball=nltk.SnowballStemmer(language='english')
print()
porter=[porter.stem(t)for t in tokens]
print('porter',porter)
print()
lan=[lancaster.stem(t)for t in tokens]
print('lancaster',lan)
print()
snowball=[snowball.stem(t)for t in tokens]
print('snowball',snowball)
print()
def clean_text2(text):
    
    text = re.sub('<.*?>+', '', text)
    token=word_tokenize(text)
    lemm=[lemmatizer.lemmatize(t) for t in token]
    #token=[t for t in token if t not in stopwords.words('english')]
    snowball=nltk.SnowballStemmer(language='english')
    snb=[snowball.stem(t) for t in lemm]
    text=' '.join(snb)
    text=re.sub('via|wa|ha|tt|ve','',text)
    return text
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
train['new_text2']=train['new_text'].apply(lambda x: clean_text2(x))
test['new_text2']=test['new_text'].apply(lambda x: clean_text2(x))
'''
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 
token=word_tokenize(train['new_text'][1270])
lemm=[lemmatizer.lemmatize(t) for t in token]
print(lemm)
'''
disaster_train=train[train['target']==1]
nondisaster=train[train['target']==0]
fig,ax=plt.subplots(1,2,figsize=[30,10])
w1=WordCloud(background_color='white').generate(''.join(disaster_train['new_text2']))
ax[0].imshow(w1)
ax[0].axis('off')
ax[0].set_title('Disaster Tweets',fontsize=40);

w2=WordCloud(background_color='white').generate(''.join(nondisaster['new_text2']))
ax[1].imshow(w2)
ax[1].axis('off')
ax[1].set_title('Non_Disaster Tweets',fontsize=40);
#info extract

#Bag of Words:describe the occurance of the words within documents
#CountVectorizer():converts the text document into matrix form(count  of each token)
#tfidf Vectorizer=it also converts the text into matrix format, but here the frequent tokens are given less weightage, and rara tokens are giiven more
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
cv=CountVectorizer(token_pattern='\w{2,}',ngram_range=(1,1),analyzer='word',)
cv.fit(train['new_text2'])
#print(cv.get_feature_names())
train_vec=cv.transform(train['new_text2'])
test_vec=cv.transform(test['new_text2'])
nb=MultinomialNB()
cv_nb=cross_val_score(nb,X=train_vec,y=train['target'],scoring='f1')
cv_nb.mean()
#TFIDF

tfidf=TfidfVectorizer(analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1,2),min_df=3)
train_tfidf=tfidf.fit_transform(train['new_text2'])
test_tfidf=tfidf.transform(test['new_text2'])
#cross_val_score(lr,train_tfidf,train['target'],cv=3,scoring='f1').mean()
from sklearn.model_selection  import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train.text, train.target, test_size=0.2, random_state=22)
tfidf=TfidfVectorizer(min_df=0,max_df=0.8,use_idf=True,ngram_range=(1,1))
train_tfidf=tfidf.fit_transform(x_train)
val_tfidf=tfidf.transform(x_val)
test_tfidf=tfidf.transform(test.text)

print('tfidf_train:',train_tfidf.shape)
print('tfidf_validation:',val_tfidf.shape)
print('tfidf_test:',test_tfidf.shape)
nb = MultinomialNB().fit(train_tfidf, y_train)
pred=nb.predict(val_tfidf)
print(classification_report(y_val,pred))
print(confusion_matrix(y_val,pred))
from collections import Counter
Counter(pred)
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
submission['target']=pd.Series(nb.predict(test_tfidf))
submission.to_csv("submission11.csv", index=False)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
#Logistic
lr=LogisticRegression().fit(train_tfidf,y_train)
pred=lr.predict(val_tfidf)
print(classification_report(y_val,pred))
print(confusion_matrix(y_val,pred))
rfc=RandomForestClassifier().fit(train_tfidf,y_train)
pred=rfc.predict(val_tfidf)
print(classification_report(y_val,pred))
print(confusion_matrix(y_val,pred))
xgb=XGBClassifier().fit(train_tfidf,y_train)
pred=xgb.predict(val_tfidf)
print(classification_report(y_val,pred))
print(confusion_matrix(y_val,pred))

help(XGBClassifier())
from sklearn.model_selection import GridSearchCV
xgb=XGBClassifier()
params={'max_depth':[2,3,4,6,8],'learning_rate':[0.0010,0.005,0.01,0.05,0.1,0.5],'n_estimator':[100,200,300]}
grid = GridSearchCV(estimator=xgb, param_grid=params, cv = 3, n_jobs=-1)
grid.fit(train_tfidf,y_train)
rs_xgb=XGBClassifier(learning_rate=0.5,max_depth=8,n_estimator=100).fit(train_tfidf,y_train)

pred=rs_xgb.predict(val_tfidf)
print(classification_report(pred,y_val))