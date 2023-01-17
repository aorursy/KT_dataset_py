# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')

df.head()
# Nans percentage 

(len(df)-df.count())/len(df)

del df['airline_sentiment_gold']

del df['negativereason_gold']

del df['tweet_coord']
df.head()
mood_count=df['airline_sentiment'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(mood_count.index, mood_count.values, alpha=0.8)

plt.title('Count of Moods')

plt.ylabel('Mood Count', fontsize=12)

plt.xlabel('Mood', fontsize=12)

plt.show()

for j in (mood_count.values):

    print(j/len(df)*100)
neg_reasons = df['negativereason'][df['airline_sentiment']=='negative'].value_counts()

neg_reasons
plt.figure(figsize=(25,5))

sns.barplot(neg_reasons.index, neg_reasons.values, alpha=0.8)

plt.title('Negative responses reasons')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Reason', fontsize=12)

plt.show()

df['user_timezone'].unique()
neg_timezone_count=df['user_timezone'][df['airline_sentiment']=='negative'].value_counts()[:10]

neg_timezone_count
neg_timezone_df = pd.DataFrame({'timezone':neg_timezone_count.index,'negative_count':neg_timezone_count.values})

neg_timezone_df.head(10)
for zone in neg_timezone_df['timezone']:

    if zone in ['Eastern Time (US & Canada)','Central Time (US & Canada)','Pacific Time (US & Canada)']:

        neg_timezone_df['timezone'][neg_timezone_df['timezone']==zone] = 'USA'

    if zone in ['Atlantic Time (Canada)','Mountain Time (US & Canada)']:

        neg_timezone_df['timezone'][neg_timezone_df['timezone']==zone] = 'Canada'

neg_timezone_df.head(10)
neg_timezone_df = neg_timezone_df.groupby('timezone',as_index=True,sort=False).sum()
neg_timezone_df = neg_timezone_df.head(10)

latitude = ['37.0902','0.1807','56.1304','34.0489','51.5074','64.2008','19.8968']

longtuide = ['-95.7129','-78.4678','-106.3468','-111.0937','0.1278','-149.4937','-155.5828']

neg_timezone_df['latitude'] = latitude

neg_timezone_df['longtuide'] = longtuide
neg_timezone_df.head()
neg_timezone_df['color']=neg_timezone_df['negative_count'].apply(lambda negative_count:"Black" if negative_count>=400 else

                                         "green" if negative_count>=300 and negative_count<400 else

                                         "Orange" if negative_count>=200 and negative_count<300 else

                                         "darkblue" if negative_count>=150 and negative_count<200 else

                                         "red" if negative_count>=100 and negative_count<150 else

                                         "lightblue" if negative_count>=75 and negative_count<100 else

                                         "brown" if negative_count>=50 and negative_count<75 else

                                         "grey")

neg_timezone_df['size']=neg_timezone_df['negative_count'].apply(lambda negative_count:20 if negative_count>=400 else

                                         15 if negative_count>=300 and negative_count<400 else

                                         12 if negative_count>=200 and negative_count<300 else

                                         11 if negative_count>=150 and negative_count<200 else

                                         10 if negative_count>=100 and negative_count<150 else

                                         7 if negative_count>=75 and negative_count<100 else

                                         5 if negative_count>=50 and negative_count<75 else

                                         3)



neg_timezone_df
neg_timezone_df['timezone'] = neg_timezone_df.index
import folium

m=folium.Map([56.1304,106.3468],zoom_start=1)

#location=location[0:2000]

for lat,lon,area,color,count,size in zip(neg_timezone_df['latitude'],neg_timezone_df['longtuide'],neg_timezone_df['timezone'],neg_timezone_df['color'],neg_timezone_df['negative_count'],neg_timezone_df['size']):

     folium.CircleMarker([lat, lon],

                            popup=area,

                            radius=size,

                            color='b',

                            fill=True,

                            fill_opacity=0.7,

                            fill_color=color,

                           ).add_to(m)

m

df.head()
sns.set(rc={'figure.figsize':(10,20)})

sns.catplot(x='airline_sentiment',kind='count',data=df,orient="h",hue='airline')

df['text_length'] =  list(map(lambda x:len(x),df['text']))
target_0 = df.loc[df['airline_sentiment'] == 'neutral']

target_1 = df.loc[df['airline_sentiment'] == 'positive']

target_2 = df.loc[df['airline_sentiment'] == 'negative']



sns.distplot(target_0[['text_length']], hist=False, rug=False,color='red',label='Neutral')

sns.distplot(target_1[['text_length']], hist=False, rug=True,color = 'yellow',label='positive')

sns.distplot(target_2[['text_length']], hist=False, rug=True,color='black',label='negative')



plt.show()

df['text'].head(20)
# Some initial features in text

qmarks = np.mean(df['text'].apply(lambda x: '?' in x))

exclamation = np.mean(df['text'].apply(lambda x: '!' in x))

at = np.mean(df['text'].apply(lambda x: '@' in x))

fullstop = np.mean(df['text'].apply(lambda x: '.' in x))

capital_first = np.mean(df['text'].apply(lambda x: x[0].isupper()))

capitals = np.mean(df['text'].apply(lambda x: max([y.isupper() for y in x])))

numbers = np.mean(df['text'].apply(lambda x: max([y.isdigit() for y in x])))

hashtags = np.mean(df['text'].apply(lambda x: '#' in x))



print('Tweets with question marks: {:.2f}%'.format(qmarks * 100))

print('Tweets with question hashtags: {:.2f}%'.format(hashtags * 100))

print('Tweets with exclamation marks: {:.2f}%'.format(exclamation * 100))

print('Tweets with full stops: {:.2f}%'.format(fullstop * 100))

print('Tweets with capitalised first letters: {:.2f}%'.format(capital_first * 100))

print('Tweets with capital letters: {:.2f}%'.format(capitals * 100))

print('Tweets with @: {:.2f}%'.format(at * 100))

print('Tweets with numbers: {:.2f}%'.format(numbers * 100))

df['has_question'] = df['text'].apply(lambda x: '?' in x)

df_has_question = df[df['has_question']]
df_has_question.head()
df_has_question.airline_sentiment.value_counts()
for j in (df_has_question.airline_sentiment.value_counts().values):

    print(j/len(df_has_question)*100)
sns.catplot(x='airline_sentiment',kind='count',data=df_has_question,orient="h",hue='airline_sentiment')

df_hasquestion = df[['has_question','airline_sentiment']]
df_hasquestion.head()
df_hasquestion['has_question'] = [1 if df_hasquestion['has_question'][x] == True else 0 for x in range(len(df_hasquestion['has_question']))]
df_hasquestion['airline_sentiment'] = [1 if df_hasquestion['airline_sentiment'][x] == 'positive' else 0 for x in range(len(df_hasquestion['airline_sentiment']))]
x = np.array(df_hasquestion['has_question'])

y = np.array(df_hasquestion['airline_sentiment'])
x=x.reshape(-1,1)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='liblinear',random_state=0).fit(x,y)

from scipy import stats



params = np.append(clf.intercept_,clf.coef_)

predictions = clf.predict(x)

newX = pd.DataFrame({"Constant":np.ones(len(x))}).join(pd.DataFrame(x))

MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))



var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())

sd_b = np.sqrt(var_b)

ts_b = params/ sd_b



p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)

ts_b = np.round(ts_b,3)

p_values = np.round(p_values,)

params = np.round(params,4)



myDF3 = pd.DataFrame()

myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]

print(myDF3)

df_hasquestion = df[df['has_question']==True]

df_hasnotques = df[df['has_question']==False]
print(len(df_hasquestion))

print(len(df_hasnotques))
sample1 = df_hasquestion.sample(n=1000,random_state= 1)

sample2 = df_hasnotques.sample(n=1000, random_state = 1)
print(len(sample1))

print(len(sample2))
var_a = sample1.var(ddof=1)

var_b = sample2.var(ddof=1)

var_a
var_b
#std deviation

s = np.sqrt((var_a + var_b)/2)

s
## Calculate the t-statistics

t = (sample1.mean() - sample2.mean())/(s*np.sqrt(2/1000))

## Compare with the critical t-value

#Degrees of freedom

deg_f = 2*1000 - 2



#p-value after comparison with the t 

p = 1 - stats.t.cdf(t,df=deg_f)





print("t = " + str(t))

print("p = " + str(2*p))
df['has_fullstops'] = df['text'].apply(lambda x: '.' in x)

df_has_fullstops = df[df['has_fullstops']]
for j in (df_has_fullstops.airline_sentiment.value_counts().values):

    print(j/len(df_has_fullstops)*100)
df['has_digits'] = df['text'].apply(lambda x: max([y.isdigit() for y in x]))

df_hasdigit=df[df['has_digits']]
for j in (df_hasdigit.airline_sentiment.value_counts().values):

    print(j/len(df_hasdigit)*100)
from wordcloud import WordCloud,STOPWORDS

df_x=df[df['airline_sentiment']=='negative']

words = ' '.join(df_x['text'].values)

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=3000,

                      height=2500

                     ).generate(cleaned_word)



plt.figure(1,figsize=(12, 20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)

import re

import nltk

from nltk.corpus import stopwords



def tweet_to_words(raw_tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words )) 

df['clean_text']=df['text'].apply(lambda x: tweet_to_words(x))

df.clean_text[:5]
from collections import Counter

words = (" ".join(df.clean_text)).lower().split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}

print('Most common words and weights: \n')

print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])

print('\nLeast common words and weights: ')

(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])

df['sentiment']=df['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)

corr=df.corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, annot=True)

df.head()
from sklearn.model_selection import train_test_split

train,test = train_test_split(df,test_size=0.2,random_state=42)



train.sentiment.value_counts()
pos_train = train[train['sentiment']==1]

neg_train = train[train['sentiment']==0]
print(len(pos_train))

print(len(neg_train))
pd.concat([pos_train,pos_train])
x_train = train['clean_text']

y_train = train['sentiment']

x_test = test['clean_text']

y_test = test['sentiment']
train_clean_tweet=[]

for tweet in x_train:

    train_clean_tweet.append(tweet)

test_clean_tweet=[]

for tweet in x_test:

    test_clean_tweet.append(tweet)

y = y_train
from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer(analyzer = "word")

train_features= v.fit_transform(train_clean_tweet)

test_features=v.transform(test_clean_tweet)

# See the words contained in our data

print(v.get_feature_names())
import sys

import numpy

numpy.set_printoptions(threshold=sys.maxsize)

# see the vector of the second word for example

print(train_features.toarray()[1:3

                              ])
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score

import xgboost as xgb

Classifiers = [

    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=200),

    AdaBoostClassifier(),

    GaussianNB(),

    xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

]

dense_features=train_features.toarray()

dense_test= test_features.toarray()

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(train_features,train['sentiment'])

        pred = fit.predict(test_features)

    except Exception:

        fit = classifier.fit(dense_features,train['sentiment'])

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,test['sentiment'])

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))    

train_clean_tweet=[]

for tweet in train['clean_text']:

    train_clean_tweet.append(tweet)

test_clean_tweet=[]

for tweet in test['clean_text']:

    test_clean_tweet.append(tweet)

y = train['sentiment']
v = CountVectorizer(analyzer = "word")

train_features= v.fit_transform(x_train)

test_features=v.transform(x_test)

from sklearn.metrics import average_precision_score

from sklearn.metrics import classification_report

dense_features=train_features.toarray()

dense_test= test_features.toarray()

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(train_features,y_train)

        pred = fit.predict(test_features)

    except Exception:

        fit = classifier.fit(dense_features,y_train)

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,y_test)

    average_precision = average_precision_score(pred,y_test)

    class_rep = classification_report(pred,y_test)

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

    print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

    print('classification report: '+str(class_rep))

    



from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()

class LemmaCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(LemmaCountVectorizer, self).build_analyzer()

        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))

tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 

                                     min_df=2,

                                     stop_words='english',

                                     decode_error='ignore')

train_features= tf_vectorizer.fit_transform(x_train)

test_features=tf_vectorizer.transform(x_test)

dense_features=train_features.toarray()

dense_test= test_features.toarray()

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(train_features,y_train)

        pred = fit.predict(test_features)

    except Exception:

        fit = classifier.fit(dense_features,y_train)

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,y_test)

    average_precision = average_precision_score(pred, y_test)

    class_rep = classification_report(pred,y_test)

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

    print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

    print('classification report: '+str(class_rep))





from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')

train_features= tfv.fit_transform(x_train)

test_features=tfv.transform(x_test)

dense_features=train_features.toarray()

dense_test= test_features.toarray()

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(train_features,y_train)

        pred = fit.predict(test_features)

    except Exception:

        fit = classifier.fit(dense_features,y_train)

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,y_test)

    average_precision = average_precision_score(pred, y_test)

    class_rep = classification_report(pred,y_test)

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

    print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

    print('classification report: '+str(class_rep))

tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',

             use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')

train_features= tfv.fit_transform(x_train)

test_features=tfv.transform(x_test)



dense_features=train_features.toarray()

dense_test= test_features.toarray()

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(train_features,y_train)

        pred = fit.predict(test_features)

    except Exception:

        fit = classifier.fit(dense_features,y_train)

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,y_test)

    average_precision = average_precision_score(pred, y_test)

    class_rep = classification_report(pred,y_test)

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

    print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

    print('classification report: '+str(class_rep))

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=120)

svd.fit(train_features)

xtrain_svd = svd.transform(train_features)

xvalid_svd = svd.transform(test_features)



# Scale the data obtained from SVD. Renaming variable to reuse without scaling.

scl = StandardScaler()

scl.fit(xtrain_svd)

xtrain_svd_scl = scl.transform(xtrain_svd)

xvalid_svd_scl = scl.transform(xvalid_svd)



dense_features=xtrain_svd_scl

dense_test= xvalid_svd_scl

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(xtrain_svd_scl,train['sentiment'])

        pred = fit.predict(xvalid_svd_scl)

    except Exception:

        fit = classifier.fit(dense_features,train['sentiment'])

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,test['sentiment'])

    average_precision = average_precision_score(pred, test['sentiment'])

    classification_rep = classification_report(pred,test['sentiment'])

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

    print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

    print('classification report',classification_rep)
x_test
x_test.index
model = Classifiers[7]

model.fit(xtrain_svd_scl,train['sentiment'])

pred= model.predict(xvalid_svd_scl)
pd.DataFrame({'pred':pred,'True':y_test})
from sklearn.pipeline import make_pipeline

from lime import lime_text

from lime.lime_text import LimeTextExplainer



c = make_pipeline(tfv, svd,scl,Classifiers[7])

class_names=list(['0','1'])

explainer = LimeTextExplainer(class_names=class_names)

idx = 4794

exp = explainer.explain_instance(x_test[idx], c.predict_proba, num_features=6)

exp.show_in_notebook(text=True)



idx = 14156

exp = explainer.explain_instance(x_test[idx], c.predict_proba, num_features=6, labels=(1,0))

exp.show_in_notebook(text=True)

svd = TruncatedSVD(n_components=180)

svd.fit(train_features)

xtrain_svd = svd.transform(train_features)

xvalid_svd = svd.transform(test_features)



# Scale the data obtained from SVD. Renaming variable to reuse without scaling.

scl = StandardScaler()

scl.fit(xtrain_svd)

xtrain_svd_scl = scl.transform(xtrain_svd)

xvalid_svd_scl = scl.transform(xvalid_svd)



dense_features=xtrain_svd_scl

dense_test= xvalid_svd_scl

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(xtrain_svd_scl,train['sentiment'])

        pred = fit.predict(xvalid_svd_scl)

    except Exception:

        fit = classifier.fit(dense_features,train['sentiment'])

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,test['sentiment'])

    average_precision = average_precision_score(pred, test['sentiment'])

    classification_rep = classification_report(pred,test['sentiment'])

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

    print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

    print('classification report',classification_rep)
from tqdm import tqdm

embeddings_index = {}

f = open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt')

for line in tqdm(f):

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))

from nltk import word_tokenize

from nltk.corpus import stopwords

stop_words = stopwords.words('english')



def sent2vec(s):

    words = str(s).lower()

    words = word_tokenize(words)

    words = [w for w in words if not w in stop_words]

    words = [w for w in words if w.isalpha()]

    M = []

    for w in words:

        try:

            M.append(embeddings_index[w])

        except:

            continue

    M = np.array(M)

    v = M.sum(axis=0)

    if type(v) != np.ndarray:

        return np.zeros(100)

    return v / np.sqrt((v ** 2).sum())

xtrain_glove = [sent2vec(x) for x in tqdm(train_clean_tweet)]

xvalid_glove = [sent2vec(x) for x in tqdm(test_clean_tweet)]

xtrain_glove = np.array(xtrain_glove)

xvalid_glove = np.array(xvalid_glove)

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(xtrain_glove,y_train)

        pred = fit.predict(xvalid_glove)

    except Exception:

        fit = classifier.fit(dense_features,y_train)

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,y_test)

    average_precision = average_precision_score(pred, y_test)

    classification_rep = classification_report(pred,y_test)

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

    print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

    print('classification report',classification_rep)
scl = StandardScaler()

xtrain_glove_scl = scl.fit_transform(xtrain_glove)

xvalid_glove_scl = scl.transform(xvalid_glove)

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(xtrain_glove_scl,y_train)

        pred = fit.predict(xvalid_glove_scl)

    except Exception:

        fit = classifier.fit(dense_features,y_train)

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,y_test)

    average_precision = average_precision_score(pred, y_test)

    classification_rep = classification_report(pred,y_test)

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

    print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

    print('classification report',classification_rep)
from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from keras.preprocessing import sequence

from keras.layers import SpatialDropout1D

from keras.preprocessing import text

# scale the data before any neural net:

model = Sequential()



model.add(Dense(300, input_dim=100, activation='relu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())



model.add(Dense(300, activation='relu'))

model.add(Dropout(0.3))

model.add(BatchNormalization())



model.add(Dense(2))

model.add(Activation('softmax'))



# compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam')

ytrain_enc = np_utils.to_categorical(y_train)

yvalid_enc = np_utils.to_categorical(y_test)

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto')

model.fit(xtrain_glove_scl, y=ytrain_enc, batch_size=64, epochs=50, 

          verbose=1, validation_data=(xvalid_glove_scl, yvalid_enc), callbacks=[earlystop])

y_pred = model.predict(xvalid_glove_scl, verbose=1)

y_pred_bool = np.argmax(y_pred, axis=1)



print(classification_report(y_test, y_pred_bool))

token = text.Tokenizer(num_words=None)

max_len = 70



token.fit_on_texts(list(x_train) + list(x_test))

xtrain_seq = token.texts_to_sequences(x_train)

xvalid_seq = token.texts_to_sequences(x_test)

x_train[0:2]
xtrain_seq[0:2]
max_len = 70

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)





word_index = token.word_index



embedding_matrix = np.zeros((len(word_index) + 1, 100))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

embedding_matrix


model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     100,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

model.add(SpatialDropout1D(0.3))

model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(2))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto')



model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=50, verbose=1, validation_data=(xvalid_pad, yvalid_enc),callbacks=[earlystop])

y_pred = model.predict(xvalid_pad, verbose=1)

y_pred_bool = np.argmax(y_pred, axis=1)



print(classification_report(y_test, y_pred_bool))

class_names=list(['0','1'])

explainer = LimeTextExplainer(class_names=class_names)

idx = 4794

exp = explainer.explain_instance(lsx_test[idx], model.predict, num_features=6, labels=(1,0))

exp.show_in_notebook(text=True)

train_clean_tweet[2957]
from keras.layers import Bidirectional



model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     100,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

model.add(SpatialDropout1D(0.3))

model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(2))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')



# Fit the model with early stopping callback

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=50, 

          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])

y_pred = model.predict(xvalid_pad, verbose=1)

y_pred_bool = np.argmax(y_pred, axis=1)



print(classification_report(y_test, y_pred_bool))

from keras.layers import GRU



model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     100,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

model.add(SpatialDropout1D(0.3))

model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))

model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.8))



model.add(Dense(2))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')



# Fit the model with early stopping callback

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, 

          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])

y_pred = model.predict(xvalid_pad, verbose=1)

y_pred_bool = np.argmax(y_pred, axis=1)



print(classification_report(y_test, y_pred_bool))
