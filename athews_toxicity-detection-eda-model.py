# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#import required packages

#basics

import pandas as pd 

import numpy as np



#misc

import gc

import time

import warnings



#stats

#from scipy.misc import imread

from scipy import sparse

import scipy.stats as ss



#viz

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec 

import seaborn as sns

from wordcloud import WordCloud ,STOPWORDS

from PIL import Image

import matplotlib_venn as venn



#nlp

import string

import re    #for regex

import nltk

from nltk.corpus import stopwords

import spacy

from nltk import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

# Tweet tokenizer does not split at apostophes which is what we want

from nltk.tokenize import TweetTokenizer   





#FeatureEngineering

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_is_fitted

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split



#settings

start_time=time.time()

color = sns.color_palette()

sns.set_style("dark")

eng_stopwords = set(stopwords.words("english"))

warnings.filterwarnings("ignore")



lem = WordNetLemmatizer()

tokenizer=TweetTokenizer()



%matplotlib inline
!du -l ../input/*
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import zipfile



zf = zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip') 

train = pd.read_csv(zf.open('train.csv'))
test_zf = zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip') 

test = pd.read_csv(test_zf.open('test.csv'))
#take a peak

train.head(10)
print("toxic:")

print(train[train.toxic==1].iloc[3,1])
print("severe toxic:")

print(train[train.severe_toxic==1].iloc[3,1])
columnsums=train.iloc[:,2:].sum()

#marking comments without any tags as "clean"

rowsums=train.iloc[:,2:].sum(axis=1)

train['clean']=(rowsums==0)

#count number of clean entries

train['clean'].sum()

print("Total comments = ",len(train))

print("Total clean comments = ",train['clean'].sum())

print("Total tags =",columnsums.sum())
null_check=train.isnull().sum()
null_check
test['comment_text'].fillna("unknown", inplace=True)
null_check=test.isna().sum()
null_check
x=train.iloc[:,2:].sum()

#plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.index, x.values, alpha=0.8)

plt.title("# per class")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Type ', fontsize=12)

#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
x=rowsums.value_counts()



#plot

plt.figure(figsize=(8,4))

ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[2])

plt.title("Multiple tags per comment")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('# of tags ', fontsize=12)



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
#clean comments

clean_mask=np.array(Image.open("../input/imagesforkernal/safe-zone.png"))

clean_mask=clean_mask[:,:,1]

#wordcloud for clean comments

subset=train[train.clean==True]

text=subset.comment_text.values

wc= WordCloud(background_color="black",max_words=2000,mask=clean_mask,stopwords=set(STOPWORDS))

wc.generate(" ".join(text))

plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Words frequented in Clean Comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
merge=pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])

df=merge.reset_index(drop=True)
df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

#Word count in each comment:

df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))

#Unique word count

df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))

#Letter count

df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))

#punctuation count

df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

#upper case words count

df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

#title case words count

df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

#Number of stopwords

df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

#Average length of the words

df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

#Word count percent in each comment:

df['word_unique_percent']=df['count_unique_word']*100/df['count_word']

#derived features

#Punct percent in each comment:

df['punct_percent']=df['count_punctuations']*100/df['count_word']
#seperate train and test features

train_feats=df.iloc[0:len(train),]

test_feats=df.iloc[len(train):,]

#join the tags

train_tags=train.iloc[:,2:]

train_feats=pd.concat([train_feats,train_tags],axis=1)
train_feats['count_sent'].loc[train_feats['count_sent']>10] = 10 

plt.figure(figsize=(12,6))

## sentenses

plt.subplot(121)

plt.suptitle("Are longer comments more toxic?",fontsize=20)

sns.violinplot(y='count_sent',x='clean', data=train_feats,split=True)

plt.xlabel('Clean?', fontsize=12)

plt.ylabel('# of sentences', fontsize=12)

plt.title("Number of sentences in each comment", fontsize=15)

# words

train_feats['count_word'].loc[train_feats['count_word']>200] = 200

plt.subplot(122)

sns.violinplot(y='count_word',x='clean', data=train_feats,split=True,inner="quart")

plt.xlabel('Clean?', fontsize=12)

plt.ylabel('# of words', fontsize=12)

plt.title("Number of words in each comment", fontsize=15)



plt.show()
train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200

#prep for split violin plots

#For the desired plots , the data must be in long format

temp_df = pd.melt(train_feats, value_vars=['count_word', 'count_unique_word'], id_vars='clean')

#spammers - comments with less than 40% unique words

spammers=train_feats[train_feats['word_unique_percent']<30]
plt.figure(figsize=(16,12))

plt.suptitle("What's so unique ?",fontsize=20)

gridspec.GridSpec(2,2)

plt.subplot2grid((2,2),(0,0))

sns.violinplot(x='variable', y='value', hue='clean', data=temp_df,split=True,inner='quartile')

plt.title("Absolute wordcount and unique words count")

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Count', fontsize=12)



plt.subplot2grid((2,2),(0,1))

plt.title("Percentage of unique words of total words in comment")

#sns.boxplot(x='clean', y='word_unique_percent', data=train_feats)

ax=sns.kdeplot(train_feats[train_feats.clean == 0].word_unique_percent, label="Bad",shade=True,color='r')

ax=sns.kdeplot(train_feats[train_feats.clean == 1].word_unique_percent, label="Clean")

plt.legend()

plt.ylabel('Number of occurances', fontsize=12)

plt.xlabel('Percent unique words', fontsize=12)



x=spammers.iloc[:,-7:].sum()

plt.subplot2grid((2,2),(1,0),colspan=2)

plt.title("Count of comments with low(<30%) unique words",fontsize=15)

ax=sns.barplot(x=x.index, y=x.values,color=color[3])



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.xlabel('Threat class', fontsize=12)

plt.ylabel('# of comments', fontsize=12)

plt.show()

print("Clean Spam example:")

print(spammers[spammers.clean==1].comment_text.iloc[1])

print("Toxic Spam example:")

print(spammers[spammers.toxic==1].comment_text.iloc[2])
#https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view

# Aphost lookup dict

APPO = {

"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not",

"tryin'":"trying"

}
def clean(comment):

    """

    This function receives comments and returns clean word-list

    """

    #Convert to lower case , so that Hi and hi are the same

    comment=comment.lower()

    #remove \n

    comment=re.sub("\\n","",comment)

    # remove leaky elements like ip,user

    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)

    #removing usernames

    comment=re.sub("\[\[.*\]","",comment)

    

    #Split the sentences into words

    words=tokenizer.tokenize(comment)

    

    # (')aphostophe  replacement (ie)   you're --> you are  

    # ( basic dictionary lookup : master dictionary present in a hidden block of code)

    words=[APPO[word] if word in APPO else word for word in words]

    words=[lem.lemmatize(word, "v") for word in words]

    words = [w for w in words if not w in eng_stopwords]

    

    clean_sent=" ".join(words)

    # remove any non alphanum,digit character

    #clean_sent=re.sub("\W+"," ",clean_sent)

    #clean_sent=re.sub("  "," ",clean_sent)

    return(clean_sent)
corpus=merge.comment_text
clean(corpus.iloc[12235])
clean_corpus=corpus.apply(lambda x :clean(x))



end_time=time.time()

print("total time till Cleaning",end_time-start_time)
### Unigrams -- TF-IDF 

# using settings recommended here for TF-IDF -- https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle



#some detailed description of the parameters

# min_df=10 --- ignore terms that appear lesser than 10 times 

# max_features=None  --- Create as many words as present in the text corpus

    # changing max_features to 10k for memmory issues

# analyzer='word'  --- Create features from words (alternatively char can also be used)

# ngram_range=(1,1)  --- Use only one word at a time (unigrams)

# strip_accents='unicode' -- removes accents

# use_idf=1,smooth_idf=1 --- enable IDF

# sublinear_tf=1   --- Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)





#temp settings to min=200 to facilitate top features section to run in kernals

#change back to min=10 to get better results

start_unigrams=time.time()

tfv = TfidfVectorizer(min_df=200,  max_features=10000, 

            strip_accents='unicode', analyzer='word',ngram_range=(1,1),

            use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')

tfv.fit(clean_corpus)

features = np.array(tfv.get_feature_names())



train_unigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])

test_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
#https://buhrmann.github.io/tfidf-analysis.html

def top_tfidf_feats(row, features, top_n=25):

    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''

    topn_ids = np.argsort(row)[::-1][:top_n]

    top_feats = [(features[i], row[i]) for i in topn_ids]

    df = pd.DataFrame(top_feats)

    df.columns = ['feature', 'tfidf']

    return df



def top_feats_in_doc(Xtr, features, row_id, top_n=25):

    ''' Top tfidf features in specific document (matrix row) '''

    row = np.squeeze(Xtr[row_id].toarray())

    return top_tfidf_feats(row, features, top_n)



def top_mean_feats(Xtr, features, grp_ids, min_tfidf=0.1, top_n=25):

    ''' Return the top n features that on average are most important amongst documents in rows

        indentified by indices in grp_ids. '''

    

    D = Xtr[grp_ids].toarray()



    D[D < min_tfidf] = 0

    tfidf_means = np.mean(D, axis=0)

    return top_tfidf_feats(tfidf_means, features, top_n)



# modified for multilabel milticlass

def top_feats_by_class(Xtr, features, min_tfidf=0.1, top_n=20):

    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value

        calculated across documents with the same class label. '''

    dfs = []

    cols=train_tags.columns

    for col in cols:

        ids = train_tags.index[train_tags[col]==1]

        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)

        feats_df.label = label

        dfs.append(feats_df)

    return dfs
train_tags.columns
tfidf_top_n_per_lass=top_feats_by_class(train_unigrams,features)



end_unigrams=time.time()



print("total time in unigrams",end_unigrams-start_unigrams)

print("total time till unigrams",end_unigrams-start_time)
plt.figure(figsize=(16,22))

plt.suptitle("TF_IDF Top words per class(unigrams)",fontsize=20)

gridspec.GridSpec(4,2)

plt.subplot2grid((4,2),(0,0))

sns.barplot(tfidf_top_n_per_lass[0].feature.iloc[0:9],tfidf_top_n_per_lass[0].tfidf.iloc[0:9],color=color[0])

plt.title("class : Toxic",fontsize=15)

plt.xlabel('Word', fontsize=12)

plt.ylabel('TF-IDF score', fontsize=12)



plt.subplot2grid((4,2),(0,1))

sns.barplot(tfidf_top_n_per_lass[1].feature.iloc[0:9],tfidf_top_n_per_lass[1].tfidf.iloc[0:9],color=color[1])

plt.title("class : Severe toxic",fontsize=15)

plt.xlabel('Word', fontsize=12)

plt.ylabel('TF-IDF score', fontsize=12)

#temp settings to min=150 to facilitate top features section to run in kernals

#change back to min=10 to get better results

tfv = TfidfVectorizer(min_df=150,  max_features=30000, 

            strip_accents='unicode', analyzer='word',ngram_range=(2,2),

            use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



tfv.fit(clean_corpus)

features = np.array(tfv.get_feature_names())

train_bigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])

test_bigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])

#get top n for bigrams

tfidf_top_n_per_lass=top_feats_by_class(train_bigrams,features)
plt.figure(figsize=(16,22))

plt.suptitle("TF_IDF Top words per class(Bigrams)",fontsize=20)

gridspec.GridSpec(4,2)

plt.subplot2grid((4,2),(0,0))

sns.barplot(tfidf_top_n_per_lass[0].feature.iloc[0:5],tfidf_top_n_per_lass[0].tfidf.iloc[0:5],color=color[0])

plt.title("class : Toxic",fontsize=15)

plt.xlabel('Word', fontsize=12)

plt.ylabel('TF-IDF score', fontsize=12)



plt.subplot2grid((4,2),(0,1))

sns.barplot(tfidf_top_n_per_lass[1].feature.iloc[0:5],tfidf_top_n_per_lass[1].tfidf.iloc[0:5],color=color[1])

plt.title("class : Severe toxic",fontsize=15)

plt.xlabel('Word', fontsize=12)

plt.ylabel('TF-IDF score', fontsize=12)

class Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, dual=False, n_jobs=1):

        self.C = C

        self.dual = dual

        self.n_jobs = n_jobs



    def predict(self, x):

        # Verify that model has been fit

        check_is_fitted(self, ['_r', '_clf'])

        return self._clf.predict(x.multiply(self._r))



    def predict_proba(self, x):

        # Verify that model has been fit

        check_is_fitted(self, ['_r', '_clf'])

        return self._clf.predict_proba(x.multiply(self._r))



    def fit(self, x, y):

        # Check that X and y have correct shape

        y = y.values

        x, y = check_X_y(x, y, accept_sparse=True)



        def pr(x, y_i, y):

            p = x[y==y_i].sum(0)

            return (p+1) / ((y==y_i).sum()+1)



        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))

        x_nb = x.multiply(self._r)

        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)

        return self

    

# model = NbSvmClassifier(C=4, dual=True, n_jobs=-1)
SELECTED_COLS=['count_sent', 'count_word', 'count_unique_word',

       'count_letters', 'count_punctuations', 'count_words_upper',

       'count_words_title', 'count_stopwords', 'mean_word_len',

       'word_unique_percent', 'punct_percent']

target_x=train_feats[SELECTED_COLS]

# target_x



TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

target_y=train_tags[TARGET_COLS]
TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print("Using only Indirect features")

model = LogisticRegression(C=3) #Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

X_train, X_valid, y_train, y_valid = train_test_split(target_x, target_y, test_size=0.33, random_state=2019)

train_loss = []

valid_loss = []

importance=[]
X_train.shape
len(y_train)
X_valid.shape
preds_train = np.zeros((y_train.shape[0],y_train.shape[1]))

preds_valid = np.zeros((y_valid.shape[0],y_valid.shape[1]))
for i, j in enumerate(TARGET_COLS):

    print('Class:= '+j)

    model.fit(X_train,y_train[j])

    preds_valid[:,i] = model.predict_proba(X_valid)[:,1]

    preds_train[:,i] = model.predict_proba(X_train)[:,1]

    train_loss_class=log_loss(y_train[j],preds_train[:,i])

    valid_loss_class=log_loss(y_valid[j],preds_valid[:,i])

    print('Trainloss=log loss:', train_loss_class)

    print('Validloss=log loss:', valid_loss_class)

    importance.append(model.coef_)

    train_loss.append(train_loss_class)

    valid_loss.append(valid_loss_class)

print('mean column-wise log loss:Train dataset', np.mean(train_loss))

print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))



end_time=time.time()

print("total time till Indirect feat model",end_time-start_time)
importance[0][0]
plt.figure(figsize=(16,22))

plt.suptitle("Feature importance for indirect features",fontsize=20)

gridspec.GridSpec(3,2)

plt.subplots_adjust(hspace=0.4)

plt.subplot2grid((3,2),(0,0))

sns.barplot(SELECTED_COLS,importance[0][0],color=color[0])

plt.title("class : Toxic",fontsize=15)

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Importance', fontsize=12)



plt.subplot2grid((3,2),(0,1))

sns.barplot(SELECTED_COLS,importance[1][0],color=color[1])

plt.title("class : Severe toxic",fontsize=15)

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Importance', fontsize=12)



plt.subplot2grid((3,2),(1,0))

sns.barplot(SELECTED_COLS,importance[2][0],color=color[2])

plt.title("class : Obscene",fontsize=15)

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Importance', fontsize=12)







plt.subplot2grid((3,2),(1,1))

sns.barplot(SELECTED_COLS,importance[3][0],color=color[3])

plt.title("class : Threat",fontsize=15)

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Importance', fontsize=12)





plt.subplot2grid((3,2),(2,0))

sns.barplot(SELECTED_COLS,importance[4][0],color=color[4])

plt.title("class : Insult",fontsize=15)

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Importance', fontsize=12)





plt.subplot2grid((3,2),(2,1))

sns.barplot(SELECTED_COLS,importance[5][0],color=color[5])

plt.title("class : Identity hate",fontsize=15)

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Importance', fontsize=12)



plt.subplot2grid((4,2),(3,0),colspan=2)

sns.barplot(SELECTED_COLS,importance[6][0],color=color[0])

plt.title("class : Clean",fontsize=15)

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

plt.xlabel('Feature', fontsize=12)

plt.ylabel('Importance', fontsize=12)



plt.show()
from scipy.sparse import csr_matrix, hstack



#Using all direct features

print("Using all features")

target_x = hstack((train_bigrams,train_unigrams,train_feats[SELECTED_COLS])).tocsr()





end_time=time.time()

print("total time till Sparse mat creation",end_time-start_time)
model = Classifier(C=4, dual=True, n_jobs=-1)

X_train, X_valid, y_train, y_valid = train_test_split(target_x, target_y, test_size=0.33, random_state=2018)

train_loss = []

valid_loss = []

preds_train = np.zeros((y_train.shape[0],y_train.shape[1]))

preds_valid = np.zeros((y_valid.shape[0],y_valid.shape[1]))

for i, j in enumerate(TARGET_COLS):

    print('Class:= '+j)

    model.fit(X_train,y_train[j])

    preds_valid[:,i] = model.predict_proba(X_valid)[:,1]

    preds_train[:,i] = model.predict_proba(X_train)[:,1]

    train_loss_class=log_loss(y_train[j],preds_train[:,i])

    valid_loss_class=log_loss(y_valid[j],preds_valid[:,i])

    print(metrics.classification_report(y_train[j], (preds_train[:,i] > 0.5).astype(int), digits=3)) 

    print('Trainloss=log loss:', train_loss_class)

    print('Validloss=log loss:', valid_loss_class)

    train_loss.append(train_loss_class)

    valid_loss.append(valid_loss_class)

print('mean column-wise log loss:Train dataset', np.mean(train_loss))

print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))





end_time=time.time()

print("total time till NB base model creation",end_time-start_time)
test_feats = test_feats.fillna(0)
test_feats.isnull().sum()
X_test = hstack((test_bigrams,test_unigrams,test_feats[SELECTED_COLS])).tocsr()
submission_zf = zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip') 

submission = pd.read_csv(submission_zf.open('sample_submission.csv'))
submission.shape
X_test_df = pd.DataFrame(X_test.toarray())
X_test.shape
preds_test = np.zeros((X_test.shape[0],6))
preds_test.shape
for i, j in enumerate(TARGET_COLS):

    print('Class:= '+j)

    preds_test[:,i] = model.predict_proba(X_test)[:,1]
submid = pd.DataFrame({'id': submission["id"]})

submission = pd.concat([submid, pd.DataFrame(preds_test, columns = TARGET_COLS)], axis=1)
submission.head(100)
submission.to_csv('submission.csv', index=False)
!pip install transformers
import tensorflow as tf