import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
from collections import Counter
import seaborn as sns
#Read data
data = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
data.columns
data["sentiment"].value_counts()
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantxt = re.sub(cleanr, ' ', sentence)
    return cleantxt

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return cleaned

sno = nltk.stem.SnowballStemmer("english")
stop = set(stopwords.words("english"))
all_positive_words = []
all_negative_words = []
final_string = []
str1 = ''
i = 0
for string in data["review"].values:
    filtered_sentence = []
    # Removes html tags from every review
    sent = cleanHtml(string)
    for w in sent.split():
        # For every word in a review clean punctions
        for cleanwords in cleanpunc(w).split():
            # if cleaned is alphabet and length og words greater than 2 then proceed
            if ((cleanwords.isalpha()) and len(cleanwords)>2):
                # check weather word is stop word or not
                if cleanwords.lower() not in stop:
                    # If word is not stop word then append it to filtered sentence
                    s = (sno.stem(cleanwords.lower())).encode('utf-8')
                    filtered_sentence.append(s)
                    if (data["sentiment"].values)[i].lower() == "positive":
                        all_positive_words.append(s)
                    if (data["sentiment"].values)[i].lower() == "negative":
                        all_negative_words.append(s)
                else:
                    continue
            else:
                continue
    # filtered_sentence is list contains all words of a review after preprocessing
    # join every word in a list to get a string format of the review
    str1 = b" ".join(filtered_sentence)
    #append all the string(cleaned reviews)to final_string
    final_string.append(str1)
    i += 1        
fig, axis = plt.subplots(1, 2)
print(len(all_positive_words))
pos_words_freq = list(Counter(all_positive_words).values())
print(len(all_negative_words))
neg_words_freq = list(Counter(all_negative_words).values())
sns.distplot(pos_words_freq, ax = axis[0])
sns.distplot(neg_words_freq, ax = axis[1])
fig.show()
data["review"] = final_string
def conv_label(label):
    if label.lower() == "positive":
        return 1
    elif label.lower() == "negative":
        return 0

data["sentiment"] = data["sentiment"].map(conv_label)
data.head(10)
freq_pos_words = nltk.FreqDist(all_positive_words)
freq_neg_words = nltk.FreqDist(all_negative_words)
freq_pos_words.most_common(15)
freq_neg_words.most_common(15)
#Bag of words vector with bi-grams
count_vect = CountVectorizer(ngram_range = (1, 2))
count_vect = count_vect.fit(data["review"].values)
bigram_wrds = count_vect.transform(data["review"].values)
#TF-Idf vector using bi-grams
count_vect_tfidf = TfidfVectorizer(ngram_range = (1, 2))
count_vect_tfidf = count_vect_tfidf.fit(data["review"].values)
tfidf_wrds  = count_vect_tfidf.transform(data["review"].values)
bigram_wrds
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
# change X to bigram_wrds to run classifier on Bag Of Words(BoW)
X = bigram_wrds
# X = tfidf_wrds
Y = data["sentiment"]
x_l, x_test, y_l, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha = 0.7)
clf.fit(x_l, y_l)
pred = clf.predict(x_test)
acc = accuracy_score(y_test, pred, normalize = True) * float(100)  
print("acc is on test data:", acc)
sns.heatmap(confusion_matrix(y_test, pred), annot = True, fmt = 'd')
train_acc = accuracy_score(y_l, clf.predict(x_l), normalize = True) * float(100)
print("train accuracy is:", train_acc)
print(classification_report(y_test, pred))
review = ["This is a worst movie","This is a good movie"]
#initialize BOW vectorizer
#we already fitted the model for train data on "count_vect"(means alredy found probabilities for train data)
vectorize = CountVectorizer(vocabulary = count_vect.vocabulary_)
#Use classifier we trained using Bag of words
polarity = clf.predict(vectorize.transform(review))
# count_vect_tfidf.transform(review)
print(polarity)

import pickle as pkl
f = open('classifier.pickle', 'wb')
pkl.dump(clf, f)
f.close()
import pickle as pkl
f = open('vectorizer.pickle', 'wb')
pkl.dump(count_vect, f)
f.close()
review = "You can take from user input"
with open("classifier.pickle", 'rb') as f:
    classifier = pkl.load(f)
with open("vectorizer.pickle", 'rb') as f:
    vectorizer = pkl.load(f)
from sklearn.feature_extraction.text import CountVectorizer
#vectorize your review which you used in training
vector_review = CountVectorizer(vocabulary = vectorizer.vocabulary_)
vector_review = vector_review.transform(review)
#predict the vectorized review using your classifier 
predict = classifier.predict(vector_review)
print(predict)