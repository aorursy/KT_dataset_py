#Öncelikle kullanacağımız python kütüphanaelerini ekliyerek çalışmamıza başlıyoruz
import numpy as np #Bu kütüphane lineer cebir için kullandığımız kütüphane fonksiyonlarını içeriyor
import pandas as pd # verilerimizi işlemek için pandas kütüphanasini kullanıyoruz(örn pd.read_scv)
from sklearn.model_selection import train_test_split #Bu işlem ile verilerimizi eğitim ve test(%70-%30) olacak çekilde bölüyoruz

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline

from subprocess import check_output 
veri = pd.read_csv('../input/Sentiment.csv')
#Veri setinde sadece ihtiyacım olacak sütunlar kalıyor.
veri = veri[['text', 'sentiment']]

#Veri Setimi Eğitim ve test verilerine ayırıyorum
train, test = train_test_split(veri, test_size = 0.1)
#Sonra Veri seti içerisindeki Nötr duyguları çıkarıyorum
train = train[train.sentiment !="Neutral"]
train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Pozirif Kelimeler")
wordcloud_draw(train_pos,'white')
print("Negatif Kelimeler")
wordcloud_draw(train_neg)
tweets =[]
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                    if 'http' not in word
                    and not word.startswith('@')
                    and not word.startswith('#')
                    and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_cleaned, row.sentiment))
    
test_pos = test[test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg = test_neg['text']
#Kelimelerin özelliklerini çıkarıyorum.
def get_words_in_tweets(tweets):
    all =[]
    for (words, sentiment) in tweets:
        all.extend(words)
    return all
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['containts(%s)' % word] = (word in document_words)
    return features
wordcloud_draw(w_features)
wordcloud_draw(w_features)
#Naive Bayes sınıflandırıcıyı eğitiyorum
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
neg_cnt = 0
pos_cnt = 0
for obj in test_neg:
    res = classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'):
        neg_cnt = neg_cnt + 1
for obj in test_pos:
    res = classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'):
        pos_cnt = pos_cnt + 1

print('[Negative]: %s/%s' % (len(test_neg), neg_cnt))
print('[Positive]: %s/%s' % (len(test_pos), pos_cnt))
