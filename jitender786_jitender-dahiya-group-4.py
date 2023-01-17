# !pip uninstall nltk



import os
wd = '/kaggle/working'
os.chdir(wd)

import pandas as pd
import xml.etree.ElementTree as ET
import re

import nltk 

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

import emoji

import string

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords
", ".join(stopwords.words('english'))

stopwords_list_570 = []
with open('/kaggle/input/stopword/stopwords_en.txt') as f:
    stopwords_list_570 = f.read().splitlines()

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')



os.chdir('/kaggle/input/train-test-data-csv/')

train_labels = pd.read_csv('train_labels.csv')

test = pd.read_csv('test.csv')


## test lables 

test_label = pd.read_csv('/kaggle/input/test-label/test_labels.csv')

test_label.head()

os.chdir('/kaggle/input/fit5149-2020-s1/data')


file_name = []
for each in os.listdir():
    file_name.append(each)

dox = []
doc_id = []

for f in file_name:
    tree = ET.parse(f)
    root = tree.getroot()
    txt = ''
    for d in root.iter('document'):
        txt += " "+d.text
#  
    dox.append(txt)
    doc_id.append(re.sub(r'.xml','',f))
    


data = pd.DataFrame()

data['id'] = doc_id

data['doc']= dox

data.shape
### removind URL 
data['doc']= data["doc"].apply(lambda s: re.sub(r'https\S+.*?\s','',s)) 

data['doc']= data["doc"].apply(lambda s: re.sub(r'@[A-Za-z].*? ','',s))



# lower case 

data['doc'] = data['doc'].apply(lambda s: s.lower())

# Removal of Punctuations

"!"#$%&\'()*+,-\./:;<=>\?@[\\]^_{|}~*"

def cleaner(txt):
    txt = re.sub(r"[!\*\.\?&'#$:;,/%^\(\)+=<>@\"…-]",' ',txt)
    return txt
    
data['doc']= data["doc"].apply(lambda s: cleaner(s))






### Removal of stopwords

STOPWORDS = set(stopwords.words('english'))

data["doc"] = data["doc"].apply(lambda s: " ".join([word for word in str(s).split() if word not in STOPWORDS]))


### Stemming

stemmer = PorterStemmer()

data['doc'] = data['doc'].apply(lambda s: " ".join([stemmer.stem(word) for word in s.split()]))




data['doc'][3]

## Lemmatization


from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

data["doc"] = data["doc"].apply(lambda x: lemmatize_words(x))

## Removal of Emojis

def convert_emojis(text):
    
    return " ".join([ " ".join(re.sub('[:]',' ',e).split()) for e in emoji.demojize(text).split()])

data["doc"] = data["doc"].apply(lambda s: convert_emojis(s))

data['doc'][3]


data.head()
train_labels.head()

df = pd.merge(data, train_labels, on='id')

df.shape


test_df = pd.merge(data , test_label, on='id')

test_df.head()

test_df.shape

d = pd.concat([df,test_df])  
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# ##frequency, 

## freq
def freq(df):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=2000, min_df=.05, max_df=.95,ngram_range=(1,2))
    v1 = vectorizer.fit_transform(df)

    v1 = v1.toarray()
    return v1
f_x_train = freq(d['doc'])
### TF–IDF

from sklearn.feature_extraction.text import TfidfVectorizer
def idf(df):
    vectorizer = TfidfVectorizer(ngram_range=(1,1),max_df=0.95,min_df=.05 )
    Xv = vectorizer.fit_transform(df)
    Xv = Xv.toarray()

    return Xv


tf_x_train = idf(d['doc'])
### one-hot

def one_hot(df):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import Binarizer
    freq = CountVectorizer(max_df=.95, min_df=.05,analyzer='word',ngram_range=(1,1))
    corpus = freq.fit_transform(df)
    onehot = Binarizer()
    corpus = onehot.fit_transform(corpus.toarray())
    return corpus

oh_x_train= one_hot(d['doc'])

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
y_train = d['gender'][0:3100]
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1700, random_state=1000,max_depth=100)
classifier.fit(oh_x_train[0:3100,], y_train) 


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

Y_pred = classifier.predict(oh_x_train[3100:,])

print(confusion_matrix(d['gender'][3100:],Y_pred))
print(classification_report(d['gender'][3100:],Y_pred))
print(accuracy_score(d['gender'][3100:], Y_pred))


classifier = RandomForestClassifier(n_estimators=3000, random_state=1000)
classifier.fit(f_x_train[0:3100,], y_train) 
Y_pred2 = classifier.predict(f_x_train[3100:,])

print(confusion_matrix(d['gender'][3100:],Y_pred2))
print(classification_report(d['gender'][3100:],Y_pred2))
print(accuracy_score(d['gender'][3100:], Y_pred2))

classifier = RandomForestClassifier(n_estimators=3000, random_state=1000)
classifier.fit(tf_x_train[0:3100,], y_train) 
Y_pred3 = classifier.predict(tf_x_train[3100:,])

print(confusion_matrix(d['gender'][3100:],Y_pred3))
print(classification_report(d['gender'][3100:],Y_pred3))
print(accuracy_score(d['gender'][3100:], Y_pred3))

Y_pred
labels = pd.DataFrame(Y_pred)

labels.to_csv()

# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Y_pred = classifier.predict(one_hot(df['doc'])[:,0:77])

# print(confusion_matrix(df['gender'],Y_pred))
# print(classification_report(df['gender'],Y_pred))
# print(accuracy_score(df['gender'], Y_pred))

# from sklearn.datasets import make_hastie_10_2
# from sklearn.ensemble import GradientBoostingClassifier

# clf = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.04 ,max_depth=100, random_state=1000).fit(oh_x_train[0:3100,], y_train)

# Y_pred4 = clf.predict(oh_x_train[3100:,])

# print(confusion_matrix(d['gender'][3100:],Y_pred4))
# print(classification_report(d['gender'][3100:],Y_pred4))
# print(accuracy_score(d['gender'][3100:], Y_pred4))

# # clf.score(test_df['gender'], Y_test)

# clf = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.1 ,max_depth=100, random_state=1000).fit(f_x_train[0:3100,], y_train)

# Y_pred5 = clf.predict(f_x_train[3100:,])

# print(confusion_matrix(d['gender'][3100:],Y_pred5))
# print(classification_report(d['gender'][3100:],Y_pred5))
# print(accuracy_score(d['gender'][3100:], Y_pred5))

# clf = GradientBoostingClassifier(n_estimators=3000, learning_rate=0.9 ,max_depth=100, random_state=1000).fit(tf_x_train[0:3100,], y_train)

# Y_pred6 = clf.predict(tf_x_train[3100:,])

# print(confusion_matrix(d['gender'][3100:],Y_pred6))
# print(classification_report(d['gender'][3100:],Y_pred6))
# print(accuracy_score(d['gender'][3100:], Y_pred6))
# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import AdaBoostClassifier

# cfr = AdaBoostClassifier(n_estimators=3000, random_state=1000)
# cfr.fit(oh_x_train[0:3100,],y_train)
# Y_pred7 = cfr.predict(oh_x_train[3100:,])


# print(confusion_matrix(d['gender'][3100:],Y_pred7))
# print(classification_report(d['gender'][3100:],Y_pred7))
# print(accuracy_score(d['gender'][3100:], Y_pred7))

# cfr = AdaBoostClassifier(n_estimators=3000, random_state=1000)
# cfr.fit(f_x_train[0:3100,],y_train)
# Y_pred8 = cfr.predict(f_x_train[3100:,])


# print(confusion_matrix(d['gender'][3100:],Y_pred8))
# print(classification_report(d['gender'][3100:],Y_pred8))
# print(accuracy_score(d['gender'][3100:], Y_pred8))


# cfr = AdaBoostClassifier(n_estimators=3000, random_state=1000)
# cfr.fit(tf_x_train[0:3100,],y_train)
# Y_pred9 = cfr.predict(tf_x_train[3100:,])


# print(confusion_matrix(d['gender'][3100:],Y_pred9))
# print(classification_report(d['gender'][3100:],Y_pred9))
# print(accuracy_score(d['gender'][3100:],Y_pred9))

# final = pd.DataFrame()
# final['id'] = test_label.iloc[:,0]


# final['gender'] = Y_pred3
# final['onehot'] = Y_pred
# final['tfidf'] = Y_pred2

# # txt = final.to_csv()

# # open('file.csv','w').write(final.to_csv())

# # f = open('file',"w+")
# # f.write(txt)


# # os.chdir('filefile')
# f = ('R','rw+')
# final.to_csv(f,sep='\t', encoding='utf-8')

# f.write(final.to_csv())

# final.to_csv('xxxx.csv',index=False)

final.to_csv("final.csv")
# test_label
# from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression(random_state=0).fit(idf(df['doc']), df['gender'])

# Y_pred10 = clf.predict(idf(test_df['doc']))


# print(confusion_matrix(d['gender'][3101:],Y_pred10))
# print(classification_report(d['gender'][3101:],Y_pred10))
# print(accuracy_score(d['gender'][3101:], Y_pred10))
