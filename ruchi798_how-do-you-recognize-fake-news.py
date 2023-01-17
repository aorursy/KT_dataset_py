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
import pandas as pd

import matplotlib.pyplot as plt

import cufflinks as cf

import plotly

import plotly.express as px

import seaborn as sns



from IPython.core.display import HTML

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer

from pandas import DataFrame

from collections import OrderedDict 

from colorama import Fore, Back, Style

y_ = Fore.YELLOW

r_ = Fore.RED

g_ = Fore.GREEN

b_ = Fore.BLUE

m_ = Fore.MAGENTA

sr_ = Style.RESET_ALL
df = pd.read_csv(r'../input/source-based-news-classification/news_articles.csv', encoding="latin", index_col=0)

df = df.dropna()

df.count()
df.head(5)
df['type'].unique()
cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
df['type'].value_counts().plot.pie(figsize = (8,8), startangle = 75)

plt.title('Types of articles', fontsize = 20)

plt.axis('off')

plt.show()
def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



def get_top_n_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]





def get_top_n_trigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
common_words = get_top_n_words(df['text_without_stopwords'], 20)

df2 = DataFrame (common_words,columns=['word','count'])

df2.groupby('word').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams used in articles',color='blue')
common_words = get_top_n_bigram(df['text_without_stopwords'], 20)

df3 = pd.DataFrame(common_words, columns = ['words' ,'count'])

df3.groupby('words').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams used in articles', color='blue')
wc = WordCloud(background_color="black", max_words=100,

               max_font_size=256,

               random_state=42, width=1000, height=1000)

wc.generate(' '.join(df['text_without_stopwords']))

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
fig = px.bar(df, x='hasImage', y='label',title='Articles including images vs Label')

fig.show()
def convert(path):

    return '<img src="'+ path + '" width="80">'
df_sources = df[['site_url','label','main_img_url']]

df_r = df_sources.loc[df['label']== 'Real'].iloc[6:10,:]

df_f = df_sources.loc[df['label']== 'Fake'].head(6)
HTML(df_r.to_html(escape=False,formatters=dict(main_img_url=convert)))
HTML(df_f.to_html(escape=False,formatters=dict(main_img_url=convert)))
df['site_url'].unique()
type_label = {'Real': 0, 'Fake': 1}

df_sources.label = [type_label[item] for item in df_sources.label] 
val_real=[]

val_fake=[]



for i,row in df_sources.iterrows():

    val = row['site_url']

    if row['label'] == 0:

        val_real.append(val)

    elif row['label']== 1:

        val_fake.append(val)
uniqueValues_real = list(OrderedDict.fromkeys(val_real)) 



print(f"{y_}Websites publishing real news:{g_}{uniqueValues_real}\n") 
uniqueValues_fake = list(OrderedDict.fromkeys(val_fake)) 

print(f"{y_}Websites publishing fake news:{r_}{uniqueValues_fake}\n")
real_set = set(uniqueValues_real) 

fake_set = set(uniqueValues_fake) 



print(f"{y_}Websites publishing both real and fake news:{m_}{real_set & fake_set}\n")
type1 = {'bias': 0, 'conspiracy': 1,'fake': 2,'bs': 3,'satire': 4, 'hate': 5,'junksci': 6, 'state': 7}

df.type = [type1[item] for item in df.type] 
def plot_bar(df, feat_x, feat_y, normalize=True):

    """ Plot with vertical bars of the requested dataframe and features"""

    

    ct = pd.crosstab(df[feat_x], df[feat_y])

    if normalize == True:

        ct = ct.div(ct.sum(axis=1), axis=0)

    return ct.plot(kind='bar', stacked=True)
plot_bar(df,'type' , 'label')

plt.show()
fig = px.sunburst(df, path=['label', 'type'])

fig.show()
df_type = df[['site_url','type']]



val_bias=[]

val_conspiracy=[]

val_fake1=[]

val_bs=[]

val_satire=[]

val_hate=[]

val_junksci=[]

val_state=[]

{'bias': 0, 'conspiracy': 1,'fake': 2,'bs': 3,'satire': 4, 'hate': 5,'junksci': 6, 'state': 7}

for i,row in df_type.iterrows():

    val = row['site_url']

    if row['type'] == 0:

        val_bias.append(val)

    elif row['type']== 1:

        val_conspiracy.append(val)

    elif row['type']== 2:

        val_fake1.append(val)

    elif row['type']== 3:

        val_bs.append(val)

    elif row['type']== 4:

        val_satire.append(val)

    elif row['type']== 5:

        val_hate.append(val)

    elif row['type']== 6:

        val_junksci.append(val)

    elif row['type']== 7:

        val_state.append(val)
uv_bias = list(OrderedDict.fromkeys(val_bias)) 

uv_conspiracy = list(OrderedDict.fromkeys(val_conspiracy)) 

uv_fake = list(OrderedDict.fromkeys(val_fake1)) 

uv_bs = list(OrderedDict.fromkeys(val_bs)) 

uv_satire = list(OrderedDict.fromkeys(val_satire)) 

uv_hate = list(OrderedDict.fromkeys(val_hate)) 

uv_junksci = list(OrderedDict.fromkeys(val_junksci)) 

uv_state = list(OrderedDict.fromkeys(val_state)) 



print(f"{b_}{type1}\n")

i=0

for lst in (uv_bias,uv_conspiracy,uv_fake,uv_bs,uv_satire, uv_hate,uv_junksci,uv_state): 

    print(f"{y_}Source URLs for type:{b_}{i}{r_}{lst}\n") 

    i+=1
df1 = df.sample(frac=1)

df1.head()
y = df1.type



x = df1.loc[:,['site_url','text_without_stopwords']]

x['source'] = x["site_url"].astype(str) +" "+ x["text_without_stopwords"] 

x = x.drop(['site_url','text_without_stopwords'],axis=1)

x = x.source
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)



tfidf_vect = TfidfVectorizer(stop_words = 'english')

tfidf_train = tfidf_vect.fit_transform(x_train)

tfidf_test = tfidf_vect.transform(x_test)

tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())
tfidf_vect
tfidf_train.shape
Adab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),n_estimators=5,random_state=1)

Adab.fit(tfidf_train, y_train)

y_pred3 = Adab.predict(tfidf_test)

ABscore = metrics.accuracy_score(y_test,y_pred3)

print("accuracy: %0.3f" %ABscore)
Rando = RandomForestClassifier(n_estimators=100,random_state=0)

Rando.fit(tfidf_train,y_train)

y_pred1 = Rando.predict(tfidf_test)

RFscore = metrics.accuracy_score(y_test,y_pred1)

print("accuracy:  %0.3f" %RFscore)