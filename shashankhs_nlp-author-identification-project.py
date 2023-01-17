import numpy as np
import pandas as pd
import nltk
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer 
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
authors = {"EAP" : "Edgar Allan Poe", "HPL" : "HP Lovecraft", "MWS" : "Mary Shelley"}
train = pd.read_csv('../input/train.csv')
train.head()
authorGrp = train.groupby('author')
text = ''
stop_words = set(nltk.corpus.stopwords.words('english'))
authorFreq = {}
new_authorFreq = {}
stop_words = stop_words.union({'I','The'})

for author, data in authorGrp:
    text = data['text'].str.cat(sep = ' ')
    tokens = []
    new_tokens = []
    tokens = nltk.word_tokenize(text)
    new_tokens = [word for word in tokens if(word.isalpha() and word not in stop_words)]
    authorFreq[author] = nltk.FreqDist(tokens)
    new_authorFreq[author] = nltk.FreqDist(new_tokens)

word = input('Enter word:')
for key in authorFreq.keys():
    print(authors[key])
    print(authorFreq[key][word])
    print('\n')
for key in new_authorFreq.keys():
    print(authors[key])
    word = new_authorFreq[key].max()
    print("Most used word: " + word + " -> " + str(new_authorFreq[key][word]))
    print('\n')
cnt_srs = train['author'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Amount of Data', fontsize=12)
plt.xlabel('Author', fontsize=12)
plt.show()
feat = train['text']
cate = train['author']
print(nltk.word_tokenize(feat[0]))
feat
vect = CountVectorizer().fit_transform(feat)
x = vect.toarray().tolist()
stop_words = set(nltk.corpus.stopwords.words('english'))
punctuations = set()
for i in string.punctuation:
    punctuations.add(i)
for i in range(len(x)):
    sw = 0
    punc = 0
    avglen = 0
    count = 0
    tokens = []
    tokens = nltk.word_tokenize(feat[i])
    for word in tokens:
        if word in punctuations:
            punc = punc + 1
        elif word in stop_words:
            sw = sw + 1
            avglen = avglen + len(word)
            count = count + 1
        else:
            avglen = avglen + len(word)
            count = count + 1
    avglen = avglen / count
    x[i].extend([sw, punc, avglen, count])
    
print(len(x))
# Divide Training and Testing data
x_train = x[:-4000]
y_train = cate[:-4000]
x_test = x[-4000:]
y_test = cate[-4000:]
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb.score(x_test, y_test)
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc.score(x_test, y_test)
svc = SVC()
svc.fit(x_train, y_train)
svc.score(x_test, y_test)