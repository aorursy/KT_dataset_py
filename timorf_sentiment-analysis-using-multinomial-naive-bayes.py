import numpy as np 
import pandas as pd 
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import emoji
import string
import re
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics

df_full=pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')
print(df_full.head(1).T)
print(
    """
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    reviewText - text of the review
    overall - rating of the product
    summary - summary of the review
    unixReviewTime - time of the review (unix time)
    reviewTime - time of the review (raw)"""
)

df_full['sentiment']=df_full.overall.apply(lambda x:1 if x>3 else 0 if x==3 else -1)
df_full=df_full.merge(df_full.groupby(by='reviewerID')[['overall']].mean(),on='reviewerID',
              suffixes=('', '_reviewer')).merge(
df_full.groupby(by='asin')[['overall']].mean(),on='asin',
              suffixes=('', '_product_mean')).merge(
df_full.groupby(by='asin')[['overall']].std(),on='asin',
              suffixes=('', '_product_std')).merge(
df_full.groupby(by='asin')[['asin']].count().rename(columns={'asin':'asin_counts'}),on='asin',
              suffixes=('', '_counts'))
df_full['upper_interval_limit']=df_full['overall_product_mean']+2.58*df_full['overall_product_std']
df_full['lower_interval_limit']=df_full['overall_product_mean']-2.58*df_full['overall_product_std']

df_full[~((df_full['overall_reviewer']>df_full['lower_interval_limit'])&
        (df_full['overall_reviewer']<df_full['upper_interval_limit']))
        &(df_full.asin_counts>3)&(df_full.overall_product_std!=0.0)]

df_norm=df_full[((df_full['overall_reviewer']>df_full['lower_interval_limit'])&
        (df_full['overall_reviewer']<df_full['upper_interval_limit']))
        ]
def check_emoji(line):
    emoji_=''.join(emoji.UNICODE_EMOJI.keys())
    emoji_flag=sum([i in emoji_ for i in line])>0
    return emoji_flag

def check_capslock(line):
    capslock_flag=len(re.findall(r'[A-Z][A-Z][A-Z]+',line))>1
    return capslock_flag

def preprocess(line):
    ps=PorterStemmer()
    remove_list=string.punctuation
    remove_list+=''.join(emoji.UNICODE_EMOJI.keys())
    translator = str.maketrans(remove_list, ' '*len(remove_list), '')
    line=line.translate(translator)
    line=re.sub(r'http(s)?:\/\/\S*? ', " ", line)
    this_stopwords=set(stopwords.words('english'))
    line = ' '.join(filter(lambda l: l not in this_stopwords, line.split(' ')))
    line=line.replace('  ','').lower()
    tokens=[]
    for word in line.split(' '):
        tokens.append(ps.stem(word))
    #line=' '.join([i if i not in stopwords.words() else '' for i in line.split(' ') ])
    
    return tokens
df_norm['summary_capslock']=df_norm.summary.apply(lambda x:check_capslock(x))
df_norm['summary_emoji']=df_norm.summary.apply(lambda x:check_emoji(x))
df_norm['summary_prep']=df_norm.summary.apply(lambda x:preprocess(x))
df_norm['reviewText_prep']=df_norm.reviewText.astype(str).apply(lambda x:preprocess(x))
df_norm['reviewText_capslock']=df_norm.reviewText.astype(str).apply(lambda x:check_capslock(x))
df_norm['reviewText_emoji']=df_norm.reviewText.astype(str).apply(lambda x:check_emoji(x))
sid = SentimentIntensityAnalyzer()
def dict_max(scores):
    if scores['pos']==max(scores.values()):
        return 1
    elif scores['neg']==max(scores.values()):
        return -1
    else:
        return 0
    
df_norm['sentiment_vader']=df_norm['summary'].astype(str).apply(lambda x:dict_max(sid.polarity_scores(x)))
print("NLTK VADER:",np.round(metrics.accuracy_score(df_norm.sentiment, df_norm.sentiment_vader),4))
cv = CountVectorizer(analyzer="word",ngram_range = (1,1))
text_counts= cv.fit_transform(df_norm['reviewText_prep'].apply(lambda x:' '.join(x)))
x_train, x_test=text_counts[:-3000],text_counts[-3000:]
y_train, y_test=df_norm.sentiment[:-3000],df_norm.sentiment[-3000:]
clf = MultinomialNB().fit(x_train, y_train)
predicted= clf.predict(x_test)
print("MultinomialNB Accuracy:",np.round(metrics.accuracy_score(y_test, predicted),4))
gnb=LinearSVC(penalty='l1',dual=False)
gnb.fit(x_train.toarray(),y_train)
predicted_svc= gnb.predict(x_test.toarray())
print("LinearSVC Accuracy:",np.round(metrics.accuracy_score(y_test, predicted_svc),4))