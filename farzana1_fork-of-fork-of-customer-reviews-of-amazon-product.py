import re, string, collections, pickle, os  #any object in Python can be pickled so that it can be saved on disk, Pickling is a way to convert a python object (list, dict, etc.) into a character stream

%matplotlib inline

import matplotlib.pyplot as plt

#import mpld3

#mpld3.enabled_notebook()

import pandas as pd

import numpy as np

import itertools #module is a collection of tools for handling iterators

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

#from sklearn.naive_bayes import MultinomialNB

from sklearn import datasets, linear_model

from sklearn.decomposition import TruncatedSVD, PCA #performs linear dimensionality reduction by means of truncated singular value decomposition (SVD).Contrary to PCA,this estimator does not center the data before computing the singular value decomposition. This means it can work with scipy.sparse matrices efficiently.Principal Component Analysis (PCA) is used to explain the variance-covariance structure of a set of variables through linear combinations. It is often used as a dimensionality-reduction technique.

from sklearn.metrics import confusion_matrix
df = pd.read_csv("../input/clean.csv",sep="|")

df.head()
# How many columns and rows

df.shape
# How many pos and neg sentence/comments in the clean.csv file

pos = df.loc[df["sentiment"]==1].copy().reset_index(drop=True)

neg = df.loc[df["sentiment"]==0].copy().reset_index(drop=True)

#neg.head()

#pos.head()

print(len(pos))

print(len(neg))
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.figure(figsize=(8, 8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
x_train,x_test,y_train,y_test = train_test_split(df["comment"].values,df["sentiment"].values,test_size=0.2)

def tokenise(s):

    return s.split(" ")

vect = CountVectorizer(tokenizer = tokenise)

tf_train = vect.fit_transform(x_train)

tf_test = vect.transform(x_test)
model = LogisticRegression(C=0.2, dual=True)

#model = MultinomialNB()

model.fit(tf_train, y_train)

preds = model.predict(tf_test)

acc = (preds==y_test).mean()

print(f'Accuracy: {acc}')

plot_confusion_matrix(confusion_matrix(y_test, preds.T), classes=['Negative', 'Positive'], title='Confusion matrix')

plt.show()
# Length of train data

len(pos), len(neg)
df = pd.read_csv("../input/data3.csv")

df.head()
#col = df["reviews.text"]

#col.head()



import re

f = open("../input/data3.csv", encoding="utf8")

text = f.read()

f.close()

text = re.sub(r",+\n","\n",text)

text = re.sub(r",0\n","|0\n",text)

text = re.sub(r",1\n","|1\n",text)

text = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_'~()\"]","",text)

#print(text)

f = open("clean1.csv","w", encoding="utf8")

f.write(text)

f.close()
import csv

df = pd.read_csv("../input/clean1.csv", quoting=csv.QUOTE_NONE)

print(df.shape)
data = (df["reviewstext"].values)

tf_data = vect.transform(data)
preds = model.predict(tf_data)

preds[preds==0] = -1

preds
main_data = pd.read_csv("../input/data2.csv")

main_data["reviews.text"][9705]

main_data = main_data.drop([3211,9705],axis=0)
main_data["sentiments"] = preds

main_data.head()
post = main_data.loc[main_data["sentiments"]==1].copy().reset_index(drop=True)

negt = main_data.loc[main_data["sentiments"]==-1].copy().reset_index(drop=True)

#negt

#post

print(len(post))

print(len(negt))
# Both negative and positive sentiments

required_data = main_data[["name","sentiments"]]

required_data.head()
# Positive Sentiments

post[["reviews.text","sentiments"]].head()
# Negative Sentiments 

negt[["reviews.text","sentiments"]].head()
# grouping the data as groupbyname

groups = required_data.groupby("name")

final = pd.DataFrame()

for name,group in groups:

    row = {}

    row["name"] =  name

    row["sentiment"] =  group["sentiments"].sum()

    final = final.append(row,ignore_index=True)

final.to_csv("../input/sentiments_final.csv")

final.head()
import matplotlib.pyplot as plt, seaborn as sns

from matplotlib import pyplot as plt

#plt.figure(figsize=(10,6))



plt.rc('xtick', labelsize=14)

plt.rc('ytick', labelsize=14)

sns.set_style(style='whitegrid')
from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

plt.figure(figsize=(14,6))

sample = final.loc[0:48]

sample.plot.bar(x="name",y="sentiment", figsize=(24, 9))

plt.ylabel("Number of Sentiments")

plt.title('Amazon Products')

plt.legend()

#plt.plot(x)

plt.show()
import matplotlib

matplotlib.style.use('ggplot')

fig = plt.figure(figsize = (15,6))

sample = final.loc[0:5]

sample.plot.bar(x="name",y="sentiment", figsize=(10, 6))

plt.ylabel("Number of Sentiments")

plt.title('Amazon Products')

plt.legend()

plt.show()
import matplotlib

matplotlib.style.use('ggplot')

fig = plt.figure(figsize = (15,6))

sample = final.loc[6:12]

sample.plot.bar(x="name",y="sentiment", figsize=(10, 6))

plt.ylabel("Number of Sentiments")

plt.title('Amazon Products')

plt.legend()

plt.show()
import matplotlib

matplotlib.style.use('ggplot')

fig = plt.figure(figsize = (15,6))

sample = final.loc[13:18]

sample.plot.bar(x="name",y="sentiment", figsize=(10, 6))

plt.ylabel("Number of Sentiments")

plt.xlabel("Name of Amazon Products")

plt.title('Amazon Products')

plt.legend()

plt.show()
import matplotlib

matplotlib.style.use('ggplot')

fig = plt.figure(figsize = (15,6))

sample = final.loc[19:25]

sample.plot.bar(x="name",y="sentiment", figsize=(10, 6))

plt.ylabel("Number of Sentiments")

plt.xlabel("Name of Amazon Produts")

plt.title('Amazon Products')

plt.legend()

plt.show()
import matplotlib

matplotlib.style.use('ggplot')

fig = plt.figure(figsize = (15,6))

sample = final.loc[0:25]

sample.plot.bar(x="name",y="sentiment", figsize=(20, 6))

plt.ylabel("Number of Sentiments")

plt.xlabel("Name of Amazon Produts")

plt.title('Amazon Products')

plt.legend()

plt.show()
sent = pd.read_csv("../input/sentiments_final.csv")

sent.head()
df = pd.read_csv("../input/data2.csv")

df.head()
#ploting graph on the basis of review ratings

df["reviews.rating"].value_counts().sort_values().plot.bar()

plt.show()
vocab = vect.get_feature_names()

len(vocab)
coef_df = pd.DataFrame({'vocab': vocab, 'coef':model.coef_.reshape(-1)})

pos_top10 = coef_df.sort_values('coef', ascending=False).reset_index(drop=True)[:10]

neg_top10 = coef_df.sort_values('coef').reset_index(drop=True)[:10]
fig, axs = plt.subplots(1, 2, figsize=(18, 10))

fig.subplots_adjust(wspace=0.8)

pos_top10.sort_values('coef').plot.barh(legend=False, ax=axs[0])

axs[0].set_yticklabels(pos_top10['vocab'].values.tolist()[::-1])

axs[0].set_title('Positive Words');

neg_top10.sort_values('coef', ascending=False).plot.barh(legend=False, ax=axs[1])

axs[1].set_yticklabels(neg_top10['vocab'].values.tolist()[::-1])

axs[1].set_title('Negative Words');
#Filtering not null values

perm = df[['reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.username']]

senti= perm[perm["reviews.rating"].notnull()]

senti.head()
#Classifying text as postive and negative¶

senti["senti"] = senti["reviews.rating"]>=4

senti["senti"] = senti["senti"].replace([True , False] , ["pos" , "neg"])
from wordcloud import WordCloud, STOPWORDS

import matplotlib as mpl

stopwords = set(STOPWORDS)

mpl.rcParams['font.size']=12                #10 

mpl.rcParams['savefig.dpi']=100             #72 

mpl.rcParams['figure.subplot.bottom']=.1 



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='lavender',

        stopwords=stopwords,

        max_words=300,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

        

    ).generate(str(data))

    

    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()

    

show_wordcloud(senti["reviews.text"])
show_wordcloud(senti["reviews.text"][senti.senti == "pos"] , title="Positive Words")
show_wordcloud(senti["reviews.text"][senti.senti == "neg"] , title="Negative words")