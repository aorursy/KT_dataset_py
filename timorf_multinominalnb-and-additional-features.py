

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import emoji
import string
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from scipy.stats import ttest_ind
plt.style.use('fivethirtyeight')
train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
def preprocess(line):
    ps=PorterStemmer()
    remove_list=string.punctuation
    remove_list+=''.join(emoji.UNICODE_EMOJI.keys())
    translator = str.maketrans(remove_list, ' '*len(remove_list), '')
    line=line.translate(translator)
    line=re.sub(r'http(s)?:\/\/\S*? ', " ", line)
    this_stopwords=set(stopwords.words('english'))
    line=re.sub('https?://\S+|www\.\S+', '', line)
    this_stopwords.add("co")
    line = ' '.join(filter(lambda l: l not in this_stopwords, line.split(' ')))
    line=line.replace("#",' ').replace('  ','').lower()
    return line
train['tokens']=train['text'].astype(str).apply(lambda x:preprocess(x).split(' '))
train['text_prep']=train['text'].astype(str).apply(lambda x:preprocess(x))
true_tokens=np.array(train.loc[train.target==1,'tokens'].apply(lambda x:np.array(x)))
false_tokens=np.array(train.loc[train.target==0,'tokens'].apply(lambda x:np.asarray(x)))
vectorizer1=CountVectorizer()
true_tokens_v=vectorizer1.fit_transform(train.loc[train.target==1,'text_prep'].astype(str).apply(lambda x:preprocess(x)))
vectorizer2=CountVectorizer()
false_tokens_v=vectorizer2.fit_transform(train.loc[train.target==0,'text_prep'].astype(str).apply(lambda x:preprocess(x)))
true_tokens_df=pd.DataFrame(columns=vectorizer1.get_feature_names(),data=true_tokens_v.toarray())
false_tokens_df=pd.DataFrame(columns=vectorizer2.get_feature_names(),data=false_tokens_v.toarray())
true_words=true_tokens_df.T[true_tokens_df.sum(axis=0)>5].sum(axis=1)
false_words=false_tokens_df.T[false_tokens_df.sum(axis=0)>5].sum(axis=1)
def check_emoji(line):
    emoji_=''.join(emoji.UNICODE_EMOJI.keys())
    emoji_flag=sum([i in emoji_ for i in line])>0
    return emoji_flag

def check_capslock(line):
    capslock_flag=len(re.findall(r'[A-Z][A-Z][A-Z]+',line))>1
    return capslock_flag

def check_url(line):
    url_flag=len(re.findall(r'http(s)?:\/\/\S*? ',line))>1
    return url_flag

sid = SentimentIntensityAnalyzer()
def dict_max(scores):
    if scores['pos']==max(scores.values()):
        return 1
    elif scores['neg']==max(scores.values()):
        return -1
    else:
        return 0
train['tags']=train['text'].apply(lambda x:x.count('#'))
train['emoji']=train['text'].apply(lambda x:float(check_emoji(x)))
train['capslock']=train['text'].apply(lambda x:float(check_capslock(x)))
train['url']=train['text'].astype(str).apply(lambda x:float(check_url(x)))
train['sentiment_vader']=train['text'].astype(str).apply(lambda x:dict_max(sid.polarity_scores(x)))
train=train.join(pd.DataFrame.from_records(train['text'].astype(str).apply(lambda x:sid.polarity_scores(x))))
fig, ax = plt.subplots()
plt.figure(figsize=(9, 3))
ax.bar(train.groupby(by='target')['emoji'].sum().index.values,
        train.groupby(by='target')['emoji'].sum().values)

ax.set_title('Emoji by target')
fig.show()
t_emoji=ttest_ind(train[train.target==0].emoji.astype('int'),
          train[train.target==1].emoji.astype('int'))
print('pvalue={}'.format(t_emoji.pvalue))
fig, ax = plt.subplots()
plt.figure(figsize=(9, 3))
ax.bar(train.groupby(by='target')['capslock'].sum().index.values,
        train.groupby(by='target')['capslock'].sum().values)
ax.set_title('Capslock by target')
fig.show()
t_emoji=ttest_ind(train[train.target==0].capslock.astype('int'),
          train[train.target==1].capslock.astype('int'))
print('pvalue={}'.format(t_emoji.pvalue))
train[['target','tags']].boxplot(by='target',figsize=(12, 4),grid=False)
t_emoji=ttest_ind(train[train.target==0].tags.astype('int'),
          train[train.target==1].tags.astype('int'))
print('pvalue={}'.format(t_emoji.pvalue))
fig, ax = plt.subplots()
ax.bar(train.groupby(by='target')['sentiment_vader'].sum().index.values,
        train.groupby(by='target')['sentiment_vader'].sum().values)
ax.set_title('Sentiment by target')
fig.show()
t_emoji=ttest_ind(train[train.target==0].sentiment_vader.astype('int'),
          train[train.target==1].sentiment_vader.astype('int'))
print('pvalue={}'.format(t_emoji.pvalue))
fig, ax = plt.subplots()
ax.bar(train.groupby(by='target')['url'].sum().index.values,
        train.groupby(by='target')['url'].sum().values)
ax.set_title('Url by target')
fig.show()
t_emoji=ttest_ind(train[train.target==0].url.astype('int'),
          train[train.target==1].url.astype('int'))
print('pvalue={}'.format(t_emoji.pvalue))
wordcloud_true = WordCloud(background_color='white',width=800,height=600,margin=0)
wordcloud_true.generate_from_frequencies(true_words.to_dict())
wordcloud_false = WordCloud(width=800,height=600,margin=0)
wordcloud_false.generate_from_frequencies(false_words.to_dict())
fig, axes = plt.subplots(1, 2)
fig.set_size_inches(16, 10, forward=True)
axes[0].imshow(wordcloud_true, interpolation='bilinear')
axes[1].imshow(wordcloud_false, interpolation='bilinear')
plt.axis("off")
plt.show()
vectorizer=CountVectorizer(train['text'].tolist(),preprocessor=preprocess,ngram_range=(1,1),min_df=2 ,max_df=0.9)
vectorizer.fit(train['text'].astype(str).apply(lambda x:preprocess(x)))
tokens_v=vectorizer.transform(train['text'].astype(str).apply(lambda x:preprocess(x)))
tokens_v_featured=np.concatenate((tokens_v.toarray(),train[['url','neu','pos','neg']].values),axis=1)
clf = MultinomialNB()
clf.fit(tokens_v[:6000],train.target[:6000])
score1=clf.score(tokens_v[6000:],train.target[6000:])
print('MultinomialNB score: {}'.format(score1))
clf_featured = MultinomialNB()
clf_featured.fit(tokens_v_featured[:6000],train.target[:6000])
score2=clf_featured.score(tokens_v_featured[6000:],train.target[6000:])
print('MultinomialNB score with features: {}'.format(score2))

rc = RidgeClassifier()
rc.fit(tokens_v[:5000],train.target[:5000])
score1=rc.score(tokens_v[5000:],train.target[5000:])
print('RidgeClassifier score: {}'.format(score1))
rc_featured = RidgeClassifier()
rc_featured.fit(tokens_v_featured[:5000],train.target[:5000])
score2=rc_featured.score(tokens_v_featured[5000:],train.target[5000:])
print('RidgeClassifier score with features: {}'.format(score2))

svc = SVC(kernel='rbf')
svc.fit(tokens_v[:5000],train.target[:5000])
score1=clf.score(tokens_v[5000:],train.target[5000:])
print('Support Vector Machine score: {}'.format(score1))
svc_featured = SVC(kernel='rbf')
svc_featured.fit(tokens_v_featured[:5000],train.target[:5000])
score2=svc_featured.score(tokens_v_featured[5000:],train.target[5000:])
print('Support Vector Machine score with features: {}'.format(score2))
bag = BaggingClassifier(base_estimator=clf,n_estimators=30)
bag.fit(tokens_v[:6000],train.target[:6000])
score1=bag.score(tokens_v[6000:],train.target[6000:])
print('BaggingClassifier score: {}'.format(score1))
bag_featured = BaggingClassifier(base_estimator=clf_featured,n_estimators=30)
bag_featured.fit(tokens_v_featured[:5000],train.target[:5000])
score2=bag_featured.score(tokens_v_featured[5000:],train.target[5000:])
print('BaggingClassifier score with features: {}'.format(score2))
submission=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

test=pd.merge(submission,test,on='id')[['id','text','target']]
test['tags']=test['text'].apply(lambda x:x.count('#'))
test['emoji']=test['text'].apply(lambda x:float(check_emoji(x)))
test['capslock']=test['text'].apply(lambda x:float(check_capslock(x)))
test['url']=test['text'].apply(lambda x:float(check_url(x)))
test['text_prep']=test['text'].astype(str).apply(lambda x:preprocess(x))
test=test.join(pd.DataFrame.from_records(test['text'].astype(str).apply(lambda x:sid.polarity_scores(x))))
#vectorizer.fit(train['text_prep'].astype(str).apply(lambda x:preprocess(x)))
tokens_v=vectorizer.transform(test['text_prep'].astype(str).apply(lambda x:preprocess(x)))
tokens_v_featured=np.concatenate((tokens_v.toarray(),test[['url','neu','pos','neg']].values),axis=1)

test.target=clf_featured.predict(tokens_v_featured)
test[['id','target']].to_csv('submission_6_nb.csv',index=False)
