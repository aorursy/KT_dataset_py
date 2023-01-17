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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.tokenize import RegexpTokenizer

import re

from textblob import TextBlob

from wordcloud import WordCloud, STOPWORDS 
TweetData =  pd.read_csv("/kaggle/input/nlp-getting-started/train.csv",index_col=0)
TweetData.head()
TweetData.info()
TweetData['tweet_length']= TweetData['text'].apply(lambda x: len(x.split()))
TweetData.head()
# Number of Sentence in a tweet

TweetData['Sent_count'] = TweetData['text'].apply(lambda x : len(sent_tokenize(x))) 
TweetData.head()
# punchuation mark count per text

TweetData['punch_count'] = TweetData['text'].apply(lambda x : len(word_tokenize(x))-len(x.split()))  
# hastags in a tweets

TweetData['hashtags'] = TweetData['text'].apply(lambda x :  word_tokenize(x).count('#'))
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 

tokenizer = RegexpTokenizer(r'\w+')
# without stopwords text length

TweetData['text_no_stop'] = TweetData['text'].apply(lambda x : len([w for w in tokenizer.tokenize(x) if not w in stop_words]))
def clean_tweets(x):

    clean1 = re.sub('https?://[A-Za-z0-9./]+','',x)

    clean2 = re.sub(r'[^\w\s]','',clean1).lower()

    return clean2
TweetData['Clean_text'] = TweetData['text'].apply(lambda x: clean_tweets(x))
TweetData['Sentiment_Score'] = TweetData['Clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
TweetData['Sentiment']="A"

Condition = [(TweetData['Sentiment_Score']>0),(TweetData['Sentiment_Score'] ==0)]

choices = ['Positive','Neutral']

TweetData['Sentiment'] = np.select(Condition,choices,default='Negative')
TweetData.head()
sns.set()

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12,4), dpi=100)

sns.distplot(TweetData['tweet_length'],ax=axes[0][0],kde=False)

sns.distplot(TweetData['text_no_stop'],ax=axes[0][1],kde=False)

sns.distplot(TweetData['Sentiment_Score'],ax=axes[1][0],kde=False)

sns.distplot(TweetData['punch_count'],ax=axes[1][1],kde=False)
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=0.9)

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6), dpi=100)

# sns.set_context('notebook',font_scale=1)

# sns.set_style('whitegrid')

sns.countplot(x='target',data=TweetData,ax=axes[0][0])

sns.countplot(x='Sent_count',data=TweetData,ax=axes[0][1])

sns.countplot(x='hashtags',data=TweetData,ax=axes[1][0])

sns.countplot(x='Sentiment',data=TweetData,ax=axes[1][1])
comment_words = ' '

stopwords = set(STOPWORDS) 
for text in TweetData['Clean_text']:

    tokens = word_tokenize(text)

    for words in tokens: 

        comment_words = comment_words + words + ' ' 
wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)

sns.set()

plt.figure(figsize = (8, 8), facecolor = None,dpi=100) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=0.9)

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6), dpi=100)

# sns.set_context('notebook',font_scale=1)

# sns.set_style('whitegrid')

sns.countplot(x='target',data=TweetData,ax=axes[0][0])

sns.countplot(x='Sent_count',data=TweetData,ax=axes[0][1],hue='target')

sns.countplot(x='hashtags',data=TweetData,ax=axes[1][0],hue='target')

sns.countplot(x='Sentiment',data=TweetData,ax=axes[1][1],hue ='target')
fig =plt.figure(figsize=(15,15),dpi=100)

sns.set_context('notebook',font_scale=1.3)

sns.set_style('whitegrid')

g=sns.pairplot(TweetData[['tweet_length','text_no_stop','Sentiment_Score','punch_count','target']],hue='target')
Tweettrain = TweetData[['target','tweet_length','Sent_count','punch_count','hashtags','text_no_stop','Sentiment_Score','Sentiment']]
X = Tweettrain.iloc[:,1:].values

Y = Tweettrain.iloc[:,0].values
from sklearn.preprocessing import LabelEncoder
LabelX = LabelEncoder()

X[:,6]=LabelX.fit_transform(X[:,6])
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
def plot_roc_curve(fpr, tpr,col,lab):

    plt.plot(fpr, tpr, color=col, label=lab)

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
X_train,X_test,Y_Train,Y_Test = train_test_split(X,Y,test_size= 0.20)
classifierRF = RandomForestClassifier(n_estimators=150,max_depth=12,criterion='gini')

classifierRF.fit(X_train,Y_Train)

y_pred_RF = classifierRF.predict(X_test)



cm_RF = confusion_matrix(Y_Test, y_pred_RF)

print('Confusion matrix Random Forest',cm_RF)



accuracy_RF = accuracy_score(Y_Test, y_pred_RF)

precision_RF =precision_score(Y_Test, y_pred_RF)

recall_RF =  recall_score(Y_Test, y_pred_RF)

f1_RF = f1_score(Y_Test, y_pred_RF)

print('accuracy random forest :',accuracy_RF)

print('precision random forest :',precision_RF)

print('recall random forest :',recall_RF)

print('f1-score random forest :',f1_RF)

auc_RF = roc_auc_score(Y_Test, y_pred_RF)

print('AUC: %.2f' % auc_RF)
xgb =  XGBClassifier(max_depth=4)

xgb.fit(X_train,Y_Train)

y_pred_xgb = xgb.predict(X_test)



cm_xgb = confusion_matrix(Y_Test, y_pred_xgb)

print('Confusion matrix Random Forest',cm_xgb)



accuracy_xgb = accuracy_score(Y_Test, y_pred_xgb)

precision_xgb =precision_score(Y_Test, y_pred_xgb)

recall_xgb =  recall_score(Y_Test, y_pred_xgb)

f1_xgb = f1_score(Y_Test, y_pred_xgb)

print('XGBOOST accuracy :',accuracy_xgb)

print('precision XGBOOST :',precision_xgb)

print('recall XGBOOST :',recall_xgb)

print('f1-score XGBOOST :',f1_xgb)

auc_xgb = roc_auc_score(Y_Test, y_pred_xgb)

print('AUC: %.2f' % auc_xgb)
classifierLogistic = LogisticRegression()

classifierLogistic.fit(X_train,Y_Train)

y_pred_logit = classifierLogistic.predict(X_test)



cm_logit = confusion_matrix(Y_Test, y_pred_logit)

print('Confusion matrix for Logistic',cm_logit)



accuracy_logit = accuracy_score(Y_Test, y_pred_logit)

precision_logit =precision_score(Y_Test, y_pred_logit)

recall_logit =  recall_score(Y_Test, y_pred_logit)

f1_logit = f1_score(Y_Test, y_pred_logit)

print('accuracy_logistic :',accuracy_logit)

print('precision_logistic :',precision_logit)

print('recall_logistic :',recall_logit)

print('f1-score_logistic :',f1_logit)

auc_logit = roc_auc_score(Y_Test, y_pred_logit)

print('AUC_logistic : %.2f' % auc_logit)
a=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

b=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

fig =plt.figure(figsize=(15,15),dpi=50)

fpr, tpr, thresholds = roc_curve(Y_Test,y_pred_logit )

plt.plot(fpr, tpr,color ='orange',label ='Logistic' )

fpr, tpr, thresholds = roc_curve(Y_Test,y_pred_RF )

plt.plot(fpr, tpr,color ='blue',label ='random Forest' )

fpr, tpr, thresholds = roc_curve(Y_Test,y_pred_RF )

plt.plot(fpr, tpr,color ='red',label ='XGB' )

plt.plot(a,b,color='black',linestyle ='dashed')

plt.legend(fontsize=15)

plt.xlabel('False Positive Rate',fontsize=15)

plt.ylabel('True Positive Rate',fontsize=15)
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv",index_col=0)
test.head()
test['tweet_length']= test['text'].apply(lambda x: len(x.split()))

test['Sent_count'] = test['text'].apply(lambda x : len(sent_tokenize(x))) 

test['punch_count'] = test['text'].apply(lambda x : len(word_tokenize(x))-len(x.split()))

test['hashtags'] = test['text'].apply(lambda x :  word_tokenize(x).count('#'))

test['text_no_stop'] = test['text'].apply(lambda x : len([w for w in tokenizer.tokenize(x) if not w in stop_words]))

test['Clean_text'] = test['text'].apply(lambda x: clean_tweets(x))

test['Sentiment_Score'] = test['Clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

test['Sentiment']="A"

Condition = [(test['Sentiment_Score']>0),(test['Sentiment_Score'] ==0)]

choices = ['Positive','Neutral']

test['Sentiment'] = np.select(Condition,choices,default='Negative')
test.head()
Test = test[['tweet_length','Sent_count','punch_count','hashtags','text_no_stop','Sentiment_Score','Sentiment']]
Test.head()
T = Test.iloc[:,:].values
T[:,6]=LabelX.transform(T[:,6])
output = classifierRF.predict(T)
outputS =  pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

outputS['target'] = output

outputS.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")