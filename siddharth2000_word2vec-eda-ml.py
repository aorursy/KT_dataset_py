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
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from prettytable import PrettyTable
df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)
df.head()
sns.countplot(df['is_sarcastic'])
df.isna().sum()

def cleaner(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", 'can not', phrase)
  
  # general
    phrase = re.sub(r"n\'t"," not", phrase)
    phrase = re.sub(r"\'re'"," are", phrase)
    phrase = re.sub(r"\'s"," is", phrase)
    phrase = re.sub(r"\'ll"," will", phrase)
    phrase = re.sub(r"\'d"," would", phrase)
    phrase = re.sub(r"\'t"," not", phrase)
    phrase = re.sub(r"\'ve"," have", phrase)
    phrase = re.sub(r"\'m"," am", phrase)
    
    return phrase
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
len(stop)
cleaned_title = []

for sentance in tqdm(df['headline'].values):
    sentance = str(sentance)
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = cleaner(sentance)
    sentance = re.sub(r'[?|!|\'|"|#|+]', r'', sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)
    cleaned_title.append(sentance.strip())
df['headline'] = cleaned_title
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize = (20,20)) # Text that is Not Sarcastic
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.is_sarcastic == 0].headline))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text that is Not Sarcastic
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.is_sarcastic == 1].headline))
plt.imshow(wc , interpolation = 'bilinear')
import plotly.offline as pyoff
import plotly.graph_objs as go
from nltk.util import ngrams
import re
import unicodedata
import nltk
def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]
sarcastic_words = basic_clean(''.join(str(df[df['is_sarcastic']==1]['headline'].tolist())))
bigram_sarcastic=(pd.Series(nltk.ngrams(sarcastic_words, 2)).value_counts())[:30]

bigram_sarcastic=pd.DataFrame(bigram_sarcastic)
bigram_sarcastic['idx']=bigram_sarcastic.index
bigram_sarcastic['idx'] = bigram_sarcastic.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+')',axis=1)
plot_data = [
    go.Bar(
        x=bigram_sarcastic['idx'],
        y=bigram_sarcastic[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'Red'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 bi-grams from Sarcastic News',
        yaxis_title='Count',
        xaxis_title='bi-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
trigram_sarcastic=(pd.Series(nltk.ngrams(sarcastic_words, 3)).value_counts())[:30]

trigram_sarcastic=pd.DataFrame(trigram_sarcastic)

trigram_sarcastic
trigram_sarcastic['idx']=trigram_sarcastic.index

trigram_sarcastic['idx'] = trigram_sarcastic.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+', ' + x['idx'][2]+')',axis=1)
plot_data = [
    go.Bar(
        x=trigram_sarcastic['idx'],
        y=trigram_sarcastic[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'Red'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 tri-grams from Sarcastic News',
        yaxis_title='Count',
        xaxis_title='tri-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
not_sarcasm = basic_clean(''.join(str(df[df['is_sarcastic']==0]['headline'].tolist())))
bigram_notsarcastic=(pd.Series(nltk.ngrams(not_sarcasm, 2)).value_counts())[:30]

bigram_notsarcastic=pd.DataFrame(bigram_notsarcastic)

bigram_notsarcastic
bigram_notsarcastic['idx']=bigram_notsarcastic.index

bigram_notsarcastic['idx'] = bigram_notsarcastic.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+')',axis=1)
plot_data = [
    go.Bar(
        x=bigram_notsarcastic['idx'],
        y=bigram_notsarcastic[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'Green'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 bi-grams from Not Sarcasm News',
        yaxis_title='Count',
        xaxis_title='bi-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
trigram_notsarcastic=(pd.Series(nltk.ngrams(not_sarcasm, 3)).value_counts())[:30]

trigram_notsarcastic=pd.DataFrame(trigram_notsarcastic)

trigram_notsarcastic['idx']=trigram_notsarcastic.index

trigram_notsarcastic['idx'] = trigram_notsarcastic.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+', ' + x['idx'][2]+')',axis=1)
plot_data = [
    go.Bar(
        x=trigram_notsarcastic['idx'],
        y=trigram_notsarcastic[0],
        #name='True',
        #x_axis="OTI",
        #y_axis="time",
        marker = dict(
            color = 'Green'
        )
    )
]
plot_layout = go.Layout(
        title='Top 30 tri-grams from Non Sarcastic News',
        yaxis_title='Count',
        xaxis_title='tri-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
import gensim
X = []
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in df['headline'].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)
w2v_model = gensim.models.Word2Vec(sentences=X, size=150, window=5, min_count=2)
w2v_model.wv.most_similar(positive = 'trump')
w2v_model.wv.most_similar(positive = 'research')
w2v_model.wv.most_similar(positive = 'study')
w2v_model.wv.similarity('trump', 'fake')
w2v_model.wv.doesnt_match(['trump', 'hillary', 'pence'])
w2v_model.wv.most_similar(positive = ['research','study'], negative=['harvard'], topn=3)

X = df['headline']
y = df['is_sarcastic']
from sklearn.model_selection import train_test_split
X_Train, X_test, y_Train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_cross, y_train, y_cross = train_test_split(X_Train, y_Train, test_size=0.1, random_state=42)
tf_idf=TfidfVectorizer(min_df=4,use_idf=True,ngram_range=(1,3))
tf_idf.fit(X_train)
Train_TFIDF = tf_idf.transform(X_train)
CrossVal_TFIDF = tf_idf.transform(X_cross)
Test_TFIDF= tf_idf.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
alpha_set=[0.0001,0.001,0.01,0.1,1,10,100,1000]

Train_AUC_TFIDF = []
CrossVal_AUC_TFIDF = []


for i in alpha_set:
    naive_b=MultinomialNB(alpha=i)
    naive_b.fit(Train_TFIDF, y_train)
    Train_y_pred =  naive_b.predict(Train_TFIDF)
    Train_AUC_TFIDF.append(roc_auc_score(y_train,Train_y_pred))
    CrossVal_y_pred =  naive_b.predict(CrossVal_TFIDF)
    CrossVal_AUC_TFIDF.append(roc_auc_score(y_cross,CrossVal_y_pred))
Alpha_set=[]
for i in range(len(alpha_set)):
    Alpha_set.append(np.math.log(alpha_set[i]))
plt.plot(Alpha_set, Train_AUC_TFIDF, label='Train AUC')
plt.scatter(Alpha_set, Train_AUC_TFIDF)
plt.plot(Alpha_set, CrossVal_AUC_TFIDF, label='CrossVal AUC')
plt.scatter(Alpha_set, CrossVal_AUC_TFIDF)
plt.legend()
plt.xlabel("alpha : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()
optimal_alpha=alpha_set[CrossVal_AUC_TFIDF.index(max(CrossVal_AUC_TFIDF))]
print(optimal_alpha)
Classifier2 = MultinomialNB(alpha=optimal_alpha)
Classifier2.fit(Train_TFIDF, y_train)
from sklearn.metrics import accuracy_score
print ("Accuracy on Train is: ", accuracy_score(y_train,Classifier2.predict(Train_TFIDF)))

print ("Accuracy on Test is: ", accuracy_score(y_test,Classifier2.predict(Test_TFIDF)))
from sklearn import metrics
print(metrics.classification_report(y_test,Classifier2.predict(Test_TFIDF)))
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(Classifier2, Test_TFIDF, y_test ,display_labels=['0','1'],cmap="Blues",values_format = '')

tf_idf=TfidfVectorizer(min_df=4,use_idf=True,ngram_range=(1,2))
tf_idf.fit(X_train)
Train_TFIDF = tf_idf.transform(X_train)
CrossVal_TFIDF = tf_idf.transform(X_cross)
Test_TFIDF= tf_idf.transform(X_test)
from sklearn.linear_model import LogisticRegression
c=[0.0001,0.001,0.01,0.1,1,10,100,1000]
Train_AUC_TFIDF = []
CrossVal_AUC_TFIDF = []
for i in c:
    logreg = LogisticRegression(C=i,penalty='l2')
    logreg.fit(Train_TFIDF, y_train)
    Train_y_pred =  logreg.predict(Train_TFIDF)
    Train_AUC_TFIDF.append(roc_auc_score(y_train ,Train_y_pred))
    CrossVal_y_pred =  logreg.predict(CrossVal_TFIDF)
    CrossVal_AUC_TFIDF.append(roc_auc_score(y_cross,CrossVal_y_pred))
C=[]
for i in range(len(c)):
    C.append(np.math.log(c[i]))
plt.plot(C, Train_AUC_TFIDF, label='Train AUC')
plt.scatter(C, Train_AUC_TFIDF)
plt.plot(C, CrossVal_AUC_TFIDF, label='CrossVal AUC')
plt.scatter(C, CrossVal_AUC_TFIDF)
plt.legend()
plt.xlabel("lambda : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()
optimal_inverse_lambda=c[CrossVal_AUC_TFIDF.index(max(CrossVal_AUC_TFIDF))]
print(pow(optimal_inverse_lambda,-1))
Classifier=LogisticRegression(C=optimal_inverse_lambda,penalty='l2')
Classifier.fit(Train_TFIDF, y_train)
print ("Accuracy on Train is: ", accuracy_score(y_train,Classifier.predict(Train_TFIDF)))
print ("Accuracy on Test is: ", accuracy_score(y_test,Classifier.predict(Test_TFIDF)))
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(Classifier, Test_TFIDF, y_test ,display_labels=['0','1'],cmap="Blues",values_format = '')
