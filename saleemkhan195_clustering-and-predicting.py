## Some of my code is inefficient, and i've probably imported things i haven't used



import nltk

from nltk.tokenize import WordPunctTokenizer

from xgboost import XGBClassifier

import pandas as pd

import numpy as np

import re

from datetime import datetime

from tqdm import tqdm

from sklearn.utils import shuffle

from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer

from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.cluster import MeanShift, KMeans

from sklearn import svm

from scipy import sparse

from scipy.sparse import coo_matrix, hstack, csr_matrix

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt



tok = WordPunctTokenizer()



df = pd.read_json('../input/Sarcasm_Headlines_Dataset.json', lines=True)

df['rowid'] = df.index



# Any results you write to the current directory are saved as output.
# I got this dictionary from somewhere online



negations_dic = {"ain't": "is not", "aren't": "are not","can't": "cannot", 

                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 

                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 

                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 

                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 

                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 

                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 

                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 

                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 

                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 

                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 

                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 

                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 

                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 

                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 

                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 

                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 

                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 

                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",

                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 

                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 

                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 

                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 

                   "this's": "this is",

                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 

                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 

                       "here's": "here is",

                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 

                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 

                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 

                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 

                   "we're": "we are", "we've": "we have", "weren't": "were not", 

                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 

                   "what's": "what is", "what've": "what have", "when's": "when is", 

                   "when've": "when have", "where'd": "where did", "where's": "where is", 

                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 

                   "who's": "who is", "who've": "who have", "why's": "why is", 

                   "why've": "why have", "will've": "will have", "won't": "will not", 

                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 

                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",

                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 

                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }



stopwords = list(STOPWORDS)



# It would be interesting if somebody applied a GA to stopwords 



stopwords.extend(["ax","i","you","edu","s","t","m","subject","can","lines","re","what",

    "there","all","we","one","the","a","an","of","or","in","for","by","on",

    "but","is","in","a","not","with","as","was","if","they","are","this","and",

    "it","have", "from","at","my","be","by","not","that","to","from","com",

    "org","like","likes","so", "hashtag", "u","thats","thing","say","really","its","looking","tho",

                 'will', 'says', 'going', 'time', 'make','still', 'next', 'got', 'every','look',

                 'actually','need','things'])

neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
lol = []

for text in tqdm(df['headline']):

    lol1 = str(text).lower()

    lol1 = neg_pattern.sub(lambda x: negations_dic[x.group()], lol1)

    lol1 = re.sub("https?:\S+", " ", lol1)

    lol1 = re.sub("@\S+", " ", lol1) # including @ handles evens out the cluster results but worsens the model 

    lol1 = re.sub("http?:\S", " ", lol1)

    lol1 = re.sub("[^a-zA-Z]", " ", lol1).strip()

    lol1 = [x for x  in tok.tokenize(lol1) if len(x) > 1]

    lol1 = (" ".join(lol1)).strip()

    lol.append(lol1)
df['headline'] = lol



tks = []

for tk in tqdm(df['headline']):

    tk = tok.tokenize(tk)

    tk = [word for word in tk if word not in stopwords]

    tk = " ".join(tk)

    tks.append(tk)

    



df['headline'] = tks
#looks like the text was cleaned properly:

df['headline'][0]
df = shuffle(df) #I don't know why either

X = df[['rowid','headline']]

y = df['is_sarcastic']

X_traindf, X_testdf, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#vectorizer = HashingVectorizer(stop_words='english',n_features=100000, non_negative=True) #This one didn't do as well

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2)) #Let bigrams be bigrams

X_train = vectorizer.fit_transform(X_traindf['headline'])

X_test = vectorizer.transform(X_testdf['headline'])
km = KMeans(n_clusters=10, n_jobs=-1, n_init=10, max_iter = 30)

kmeans = km.fit(X_train)

kmdists = km.transform(X_train)

labs = km.labels_

labstest = km.predict(X_test)



kmdiststest = km.transform(X_test)
#somehow stitch the distances from the centroids back onto the training and test data 

kmdists = coo_matrix(kmdists)

X_train2 = csr_matrix(hstack([X_train,kmdists]))

X_test2 = csr_matrix(hstack([X_test,kmdiststest]))
X_traindf['kmeanslab'] = labs

X_testdf['kmeanslab'] = labstest

X_traindf['y'] = y_train
pd.value_counts(X_traindf['kmeanslab']).plot.bar() # Never again.
clusters = sorted(pd.unique(X_traindf['kmeanslab']))

print(clusters)
#sense checking

wordcloud = []

cloud = []

for i in tqdm(clusters):

    group = X_traindf[X_traindf['kmeanslab'] == i]

    groupsize = len(group)

    sarcastic = len(group[group['y']== 1])

    not_sarcastic = len(group[group['y']== 0] )

    print("Group Size: ",groupsize)

    print("sarcasm: ", sarcastic)

    print("not sarcasm: ", not_sarcastic)

    print("sarcasm rate: ", sarcastic/groupsize)

    cloud.append(" ".join(X_traindf[X_traindf['kmeanslab'] == i]["headline"]))

    wordcloud.append(WordCloud(stopwords=stopwords, background_color='white').generate(cloud[i]))

    print("Printing wordcloud for group: ", i)

    plt.imshow(wordcloud[i], interpolation='bilinear')

    plt.axis("off")

    plt.show()
lr = LogisticRegression()

lr.fit(X_train, y_train)

lrpreds = lr.predict(X_test)

print(confusion_matrix(y_test, lrpreds))

lrscore = lr.score(X_test,y_test)

print(lrscore)
lr2 = LogisticRegression()

lr2.fit(X_train2, y_train)

lrpreds2 = lr2.predict(X_test2)

print(confusion_matrix(y_test, lrpreds2))

lr2score = lr2.score(X_test2,y_test)

print(lr2score)
sv = svm.LinearSVC()

sv.fit(X_train, y_train)

svpreds = sv.predict(X_test)

print(confusion_matrix(y_test, svpreds))

svscore = sv.score(X_test,y_test)

print(svscore)
sv2 = svm.LinearSVC()

sv2.fit(X_train2, y_train)

sv2preds = sv2.predict(X_test2)

print(confusion_matrix(y_test, sv2preds))

sv2score = sv2.score(X_test2,y_test)

print(sv2score)
gnb = MultinomialNB()

gnb.fit(X_train, y_train)

gnbpreds = gnb.predict(X_test)

print(confusion_matrix(y_test, gnbpreds))

gnbscore = gnb.score(X_test,y_test)

print(gnbscore)
gnb2 = MultinomialNB()

gnb2.fit(X_train2, y_train)

gnb2preds = gnb2.predict(X_test2)

print(confusion_matrix(y_test, gnb2preds))

gnb2score = gnb2.score(X_test2,y_test)

print(gnb2score)
sgd = SGDClassifier()

sgd.fit(X_train, y_train)

sgdpreds = sgd.predict(X_test)

print(confusion_matrix(y_test, sgdpreds))

sgdscore = sgd.score(X_test,y_test)

print(sgdscore)
sgd2 = SGDClassifier()

sgd2.fit(X_train2, y_train)

sgd2preds = sgd2.predict(X_test2)

print(confusion_matrix(y_test, sgd2preds))

sgd2score = sgd2.score(X_test2,y_test)

print(sgd2score)