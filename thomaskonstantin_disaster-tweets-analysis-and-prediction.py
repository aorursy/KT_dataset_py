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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

from sklearn.metrics import f1_score as f1

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import plotly.graph_objs as go

import plotly.express as ex
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_train.head(4)
df_train.shape[0]
df_train.isna().sum()
def remove_ht(sir):

    idx = sir.find('%20')

    if idx ==  -1:

        return sir

    else:

        return sir[0:idx]+' '+sir[(idx+3):]





df_train['keyword'].fillna(df_train['keyword'].mode()[0],inplace=True)

df_train['keyword'] = df_train['keyword'].apply(remove_ht)



df_test['keyword'].fillna(df_test['keyword'].mode()[0],inplace=True)

df_test['keyword'] = df_test['keyword'].apply(remove_ht)

def number_of_hashtags(sir):

    splited = sir.split(' ')

    ht = 0

    for word in splited:

        if len(word) > 1 and word[0] == '#':

            ht+=1

    return ht





sid = SentimentIntensityAnalyzer()



def pos_sentiment(sir):

    r = sid.polarity_scores(sir)

    return (r['pos'])

def neg_sentiment(sir):

    r = sid.polarity_scores(sir)

    return (r['neg'])



def number_of_words(sir):

    splited = sir.split(' ')

    words = 0

    for word in splited:

        if len(word) > 1 and word[0] != '#':

            words+=1

    return words

def number_of_exclamation_marks(sir):

    ex = 0

    for char in sir:

        if char == '!':

            ex+=1

    return ex



def average_word_length(sir):

    splited = sir.split(' ')

    no_hash = [word for word in splited if '#' not in word]

    length = 0

    for word in no_hash:

        length+=len(word)

    return length/len(no_hash)



def contains_mentions(sir):

    splited = sir.split(' ')

    for word in splited:

        if '@' in word:

            return 1

    return 0





disasters = ['fire','storm','flood','tornado','earthquake','volcano','hurricane',

            'tornado','cyclone','famine','epidemic','war','dead','collapse','crash','hostages','terror']



def contains_disaster_tag(sir):

    splited = sir.split(' ')

    for word in splited:

        if word.lower() in disasters:

            return 1

    return 0
df_train['Number_Of_Hashtags'] = df_train['text'].apply(number_of_hashtags)

df_train['Pos_Sentiment'] = df_train['text'].apply(pos_sentiment)

df_train['Neg_Sentiment'] = df_train['text'].apply(neg_sentiment)

df_train['Number_Of_Words'] = df_train['text'].apply(number_of_words)

df_train['Exc_Marks'] = df_train['text'].apply(number_of_exclamation_marks)

df_train['Avg_Word_Length'] = df_train['text'].apply(average_word_length)

df_train['Has_Mention'] = df_train['text'].apply(contains_mentions)

df_train['Has_Disaster_Word'] = df_train['text'].apply(contains_disaster_tag)



df_test['Number_Of_Hashtags'] = df_test['text'].apply(number_of_hashtags)

df_test['Pos_Sentiment'] = df_test['text'].apply(pos_sentiment)

df_test['Neg_Sentiment'] = df_test['text'].apply(neg_sentiment)

df_test['Number_Of_Words'] = df_test['text'].apply(number_of_words)

df_test['Exc_Marks'] = df_test['text'].apply(number_of_exclamation_marks)

df_test['Avg_Word_Length'] = df_test['text'].apply(average_word_length)

df_test['Has_Mention'] = df_test['text'].apply(contains_mentions)

df_test['Has_Disaster_Word'] = df_test['text'].apply(contains_disaster_tag)
from sklearn.preprocessing import LabelEncoder

label_e = LabelEncoder()

label_e.fit(df_train['keyword'])

df_train['Keyword'] = label_e.transform(df_train['keyword'])



label_e = LabelEncoder()

label_e.fit(df_test['keyword'])

df_test['Keyword'] = label_e.transform(df_test['keyword'])
from wordcloud import WordCloud,STOPWORDS

import re

stopwords = list(STOPWORDS)

#find top 10 words in disasters 

df_dis = df_train[df_train['target']==1]



dis_word_freq = dict()



for sample in df_dis.text:

    tokens = sample.lower()

    tokens = re.findall(r'\b[A-Za-z]+\b',tokens)

    tokens = [tok for tok in tokens if len(tok) > 2]

    no_hash = [tok for tok in tokens if '#' not in tok and tok.find('http') == -1]

    clean_tokens = [tok for tok in no_hash if tok not in stopwords]

    for tok in clean_tokens:

        if tok not in dis_word_freq:

            dis_word_freq[tok] = 1

        else:

            dis_word_freq[tok] += 1



dis_word_freq = {k: v for k, v in sorted(dis_word_freq.items(), key=lambda item: item[1])}

wl_d = list(dis_word_freq.keys())

wl_d = list(reversed(wl_d))



df_not_dis = df_train[df_train['target']==0]



not_dis_word_freq = dict()



for sample in df_not_dis.text:

    tokens = sample.lower()

    tokens = re.findall(r'\b[A-Za-z]+\b',tokens)

    tokens = [tok for tok in tokens if len(tok) > 2]

    no_hash = [tok for tok in tokens if '#' not in tok and tok.find('http') == -1]

    clean_tokens = [tok for tok in no_hash if tok not in stopwords]

    for tok in clean_tokens:

        if tok not in not_dis_word_freq:

            not_dis_word_freq[tok] = 1

        else:

            not_dis_word_freq[tok] += 1



not_dis_word_freq = {k: v for k, v in sorted(not_dis_word_freq.items(), key=lambda item: item[1])}

wl = list(not_dis_word_freq.keys())

wl = list(reversed(wl))







top_50_dist_words = wl_d[:50]

top_50_non_dist_words = wl[:50]



len(set())
def amount_of_dis_tokens(sir):

    tok = sir.split(' ')

    tok = set(tok)

    cont = set(top_50_dist_words).intersection(tok)

    return len(cont)

    

def amount_of_non_dis_tokens(sir):

    tok = sir.split(' ')

    tok = set(tok)

    cont = set(top_50_non_dist_words).intersection(tok)

    return len(cont)



df_train['Contains_Top50_Dist_Words'] = df_train.text.apply(amount_of_dis_tokens)

df_train['Contains_Top50_Non_Dist_Words'] = df_train.text.apply(amount_of_dis_tokens)



df_test['Contains_Top50_Dist_Words'] = df_test.text.apply(amount_of_dis_tokens)

df_test['Contains_Top50_Non_Dist_Words'] = df_test.text.apply(amount_of_dis_tokens)
features = df_train.columns[5:]

Y = df_train['target']





words = ''



for sample in df_train.text:

    tokens = sample.lower().split(' ')

    no_hash = [tok for tok in tokens if '#' not in tok and tok.find('http') == -1]

    words += ' '.join(no_hash)+' '



wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
plt.figure(figsize=(20,11))

ax = sns.boxplot(x=df_train['target'],y=df_train['Avg_Word_Length'])

ax.set_xlabel('Target',fontsize=18)

ax.set_ylabel('Average Word Length',fontsize=18)

ax.set_title('Average Word Length Effect On Outcome',fontsize=18)
plt.figure(figsize=(20,11))

ax = sns.kdeplot(df_train[df_train['target']==1]['Number_Of_Words'],label='Number Of Words Target=1')

ax = sns.kdeplot(df_train[df_train['target']==0]['Number_Of_Words'],label='Number Of Words Target=0')

plt.legend(prop={'size':20})
plt.figure(figsize=(20,11))

pivot = df_train.pivot_table(index='Number_Of_Words',columns='Contains_Top50_Dist_Words',values='target')

sns.heatmap(pivot,cmap='coolwarm',annot=True)

plt.figure(figsize=(20,11))

ax=sns.boxplot(x=df_train['target'],y=df_train['Neg_Sentiment'])
plt.figure(figsize=(20,11))

ax=sns.boxplot(x=df_train['target'],y=df_train['Avg_Word_Length'])
plt.figure(figsize=(20,11))

ax=sns.countplot(df_train['Contains_Top50_Dist_Words'])
df_train = df_train[df_train['Avg_Word_Length'] < 11]

df_train = df_train[df_train['Avg_Word_Length'] > 2]

df_train = df_train[df_train['Neg_Sentiment'] < 0.7]

lns = df_train[(df_train['Neg_Sentiment'] > 0.6) & (df_train['target'] == 0)]

df_train = df_train.drop(lns.index)
df_train[features]
from sklearn.feature_selection import SelectKBest,chi2

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,f1_score,classification_report

from sklearn.metrics import accuracy_score as ascore

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

 

Y = df_train['target']

selector = SelectKBest(chi2,k=4)

X = selector.fit_transform(df_train[features],Y)







train_x,test_x,train_y,test_y = train_test_split(X,Y)



s_fet = [fet for index,fet in enumerate(features) if selector.get_support()[index]==True ]

s_fet
def optimal_n(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = KNeighborsClassifier(n_neighbors = n)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(f1(pred,test_y))

    return results
n_list = [10,20,30,50,80,130,210,350,560]

result = optimal_n(train_x,test_x,train_y,test_y,n_list)

plt.figure(figsize=(20,11))

ax =sns.lineplot(x=np.arange(len(n_list)),y=result)

n_list.insert(0,1)

ax.set_xticklabels(n_list)

ax.set_title('KNN Accuracy Depending On Number Of Neighbors',fontsize=16)

ax.set_xlabel('N Value',fontsize=16)

ax.set_ylabel('Accuracy Score',fontsize=16)

plt.show()
def optimal_e(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = RandomForestClassifier(max_leaf_nodes = n,random_state=42)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(f1(pred,test_y))

    return results
n_list = [2,3,5,8,13,21,35,56,91,147,200]

result = optimal_e(train_x,test_x,train_y,test_y,n_list)

plt.figure(figsize=(20,11))

ax = sns.lineplot(x=np.arange(0,11),y=result)

#n_list.insert(0,1)

ax.set_xticklabels(labels = n_list)

ax.set_title('RandomForest Accuracy Depending On Number Of Estimators',fontsize=16)

ax.set_xlabel('N Value',fontsize=16)

ax.set_ylabel('Accuracy Score',fontsize=16)

plt.show()
def optimal_n(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = AdaBoostClassifier(n_estimators = n,random_state=42,learning_rate=0.05)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(f1(pred,test_y))

    return results
ee_list = [2,3,5,8,13,21,35,56,91,147,300]

result = optimal_n(train_x,test_x,train_y,test_y,ee_list)

plt.figure(figsize=(20,11))

ax =sns.lineplot(x=np.arange(len(ee_list)),y=result)

n_list.insert(0,1)

ax.set_xticklabels(labels = ee_list)

ax.set_title('AdaBoost Accuracy Depending On Number Of Max Leaf Nodes',fontsize=16)

ax.set_xlabel('N Value',fontsize=16)

ax.set_ylabel('Accuracy Score',fontsize=16)

plt.show()
def optimal_n(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = DecisionTreeClassifier(max_leaf_nodes = n,random_state=42,criterion='entropy')

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(f1(pred,test_y))

    return results
ee_list = [2,3,5,8,13,21,35,56,91,147,300]

result = optimal_n(train_x,test_x,train_y,test_y,ee_list)

plt.figure(figsize=(20,11))

ax =sns.lineplot(x=np.arange(len(ee_list)),y=result)

n_list.insert(0,1)

ax.set_xticklabels(labels = ee_list)

ax.set_title('Decision Tree Accuracy Depending On Number Of Max Leaf Nodes',fontsize=16)

ax.set_xlabel('N Value',fontsize=16)

ax.set_ylabel('Accuracy Score',fontsize=16)

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD

from wordcloud import STOPWORDS

import re 

import string

from nltk.stem import PorterStemmer

stop_words = list(STOPWORDS)

stemmer = PorterStemmer()

df_train.text = df_train.text.apply(lambda x : re.sub(r'^RT[\s]+', '', x))

df_train.text = df_train.text.apply(lambda x : re.sub(r'#', '', x))

df_train.text = df_train.text.apply(lambda x : re.sub(r'https?:\/\/.*[\r\n]*', '', x))

df_train.text = df_train.text.apply(lambda x : ' '.join([word for word in x.split(' ') if word not in stop_words]))

df_train.text =df_train.text.str.lower()

df_train.text = df_train.text.apply(lambda x : ' '.join([ stemmer.stem(word) for word in x.split(' ')]) )



df_train





vect = CountVectorizer()

vect.fit(df_train.text)



tf_matrix = vect.transform(df_train.text)

svd = TruncatedSVD(n_components=1500)

train_sparse = svd.fit_transform(tf_matrix)

cum_var = np.cumsum(svd.explained_variance_ratio_)

tr1 = go.Scatter(x=np.arange(len(cum_var)),y=cum_var)

go.Figure(data=[tr1],layout={'title':'explained variance ratio train data','xaxis_title':"Number Of Components",'yaxis_title':"explained variance"})
df_test.text = df_test.text.apply(lambda x : re.sub(r'^RT[\s]+', '', x))

df_test.text = df_test.text.apply(lambda x : re.sub(r'#', '', x))

df_test.text = df_test.text.apply(lambda x : re.sub(r'https?:\/\/.*[\r\n]*', '', x))

df_test.text = df_test.text.apply(lambda x : ' '.join([word for word in x.split(' ') if word not in stop_words]))

df_test.text = df_test.text.str.lower()

df_test.text = df_test.text.apply(lambda x : ' '.join([ stemmer.stem(word) for word in x.split(' ')]) )



df_test





vect = CountVectorizer()

vect.fit(df_test.text)



tf_matrix = vect.transform(df_test.text)

svd = TruncatedSVD(n_components=1500)

test_sparse = svd.fit_transform(tf_matrix)

cum_var = np.cumsum(svd.explained_variance_ratio_)

tr1 = go.Scatter(x=np.arange(len(cum_var)),y=cum_var)

go.Figure(data=[tr1],layout={'title':'explained variance ratio test data','xaxis_title':"Number Of Components",'yaxis_title':"explained variance"})
df_train
from sklearn.naive_bayes import GaussianNB

tr_sp = pd.DataFrame(train_sparse.copy())

tr_sp['Pos_Sentiment'] = df_train.Pos_Sentiment

tr_sp['Neg_Sentiment'] = df_train.Neg_Sentiment

tr_sp['Number_Of_Hashtags'] = df_train.Number_Of_Hashtags

tr_sp.Number_Of_Hashtags = tr_sp.Number_Of_Hashtags.fillna(tr_sp.Number_Of_Hashtags.mean())

tr_sp.Pos_Sentiment = tr_sp.Pos_Sentiment.fillna(tr_sp.Pos_Sentiment.mean())

tr_sp.Neg_Sentiment = tr_sp.Neg_Sentiment.fillna(tr_sp.Neg_Sentiment.mean())

train_x,test_x,train_y,test_y = train_test_split(tr_sp,df_train.target)

NB = GaussianNB()

NB.fit(train_x,train_y)

pred = NB.predict(test_x)

conf = confusion_matrix(pred,test_y)

plt.figure(figsize=(20,11))

ax = sns.heatmap(conf,annot=True,cmap='mako',fmt='d')

ax.set_title('Naive Bayes Confusion Matrix ')
from sklearn.metrics import f1_score as f1

f1(pred,test_y)
selector = SelectKBest(chi2,k=4)

X = selector.fit_transform(df_train[features],Y)

s_fet = [fet for index,fet in enumerate(features) if selector.get_support()[index]==True ]



rfc = RandomForestClassifier(max_leaf_nodes = 200,random_state=42)

dtc = DecisionTreeClassifier(max_leaf_nodes = 21,random_state=42,criterion='entropy')



rfc.fit(X,Y)

dtc.fit(X,Y)



sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

prediction = np.round(rfc.predict(df_test[s_fet])*0.5+dtc.predict(df_test[s_fet])*0.5).astype('int64')



cf_mat = (confusion_matrix(prediction,sub['target']))

plt.figure(figsize=(20,11))

ax = sns.heatmap(cf_mat,cmap='coolwarm',annot=True,fmt='d')





result = sub.copy()

result['target'] = prediction

result.to_csv('submission.csv',index=False)