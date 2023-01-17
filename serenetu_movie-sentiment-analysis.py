import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score
# Read In Data

data_set = pd.read_csv('../input/labeledTrainData.tsv', sep = '\t')
print (data_set.head())
test_set = pd.read_csv('../input/testData.tsv', sep = '\t')
print (test_set.head())

print (data_set.shape)
print (test_set.shape)
# train test split

label = data_set['sentiment']
review = data_set['review']
test = test_set['review']
print (label.head())
print (review.head())

from sklearn.model_selection import train_test_split

label_train, label_test, review_train, review_test = train_test_split(label, review, test_size = 0.2, random_state = 0)
print(label_train.shape, label_test.shape, review_train.shape, review_test.shape)
# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
review_train_count = vec.fit_transform(review_train)
review_test_count = vec.transform(review_test)
# print (vec.get_feature_names())
print (review_train_count.toarray()[:10])
# Naive Bayes From sklearn
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(review_train_count, label_train)
review_test_predict = clf.predict(review_test_count)
accuracy_score(label_test, review_test_predict)
# classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(label_test, review_test_predict))
print (confusion_matrix(label_test, review_test_predict))
# Train All Data

vec = CountVectorizer()
review_count = vec.fit_transform(review)
clf = MultinomialNB()
clf.fit(review_count, label)

test_count = vec.transform(test)
test_predict = clf.predict(test_count)

predict_df = pd.DataFrame({'id': test_set['id'], 'sentiment': test_predict})
predict_df.to_csv('../output/nb.csv', index = False, header = True)
# Achieve Naive Bayes By hands

# 1. Clean Reviews And Vectorize Them
class Review2Vector:
    def __init__(self):
        return
    
    def clean_review(self, review_raw):
        review_clean = []
        for one_review in review_raw:
            line = re.sub('[^a-zA-Z]', ' ', one_review)
            line = line.lower()
            review_clean.append(line.split())
        return np.array(review_clean)
    
    def get_word_dict(self, review_clean):
        self.word_dict = {}
        count = 0
        for one_review in review_clean:
            for word in one_review:
                if word not in self.word_dict:
                    self.word_dict[word] = count
                    count += 1
        return
    
    def get_word_matrix(self, review_clean):
        word_matrix = np.zeros((len(review_clean), len(self.word_dict)))
        for i_line, one_review in enumerate(review_clean):
            word_line = word_matrix[i_line]
            for word in one_review:
                if word in self.word_dict:
                    word_line[self.word_dict[word]] += 1
        return word_matrix
    
    def fit_transform(self, review_raw):
        review_clean = self.clean_review(review_raw)
        self.get_word_dict(review_clean)
        word_matrix = self.get_word_matrix(review_clean)
        return word_matrix
    
    def transform(self, review_raw):
        review_clean = self.clean_review(review_raw)
        word_matrix = self.get_word_matrix(review_clean)
        return word_matrix
        

r2v = Review2Vector()
review_train_count = r2v.fit_transform(review_train)
review_test_count = r2v.transform(review_test)

# 2. Hand Write NB
class MyNaiveBayes:
    def fit(self, review, label):
        assert(len(review) == len(label))
        self.word_prop_0 = np.ones(len(review[0]))
        self.word_prop_1 = np.ones(len(review[0]))
        count_0 = 0
        count_1 = 0
        for i, l in enumerate(label):
            if l == 0:
                self.word_prop_0 += review[i]
                count_0 += 1
            elif l == 1:
                self.word_prop_1 += review[i]
                count_1 += 1
        self.p0 = np.log(count_0 / (count_0 + count_1))
        self.p1 = np.log(count_1 / (count_0 + count_1))
        self.word_prop_0 = np.log(self.word_prop_0 / np.sum(self.word_prop_0))
        self.word_prop_1 = np.log(self.word_prop_1 / np.sum(self.word_prop_1))
        return
    
    def predict(self, test):
        res = []
        for one_test in test:
            p0 = np.sum(one_test * self.word_prop_0) + self.p0
            p1 = np.sum(one_test * self.word_prop_1) + self.p1
            if p0 > p1:
                res.append(0)
            else:
                res.append(1)
        return res

# Train Part
clf = MyNaiveBayes()
clf.fit(review_train_count, label_train)
review_test_predict = clf.predict(review_test_count)
print (accuracy_score(label_test, review_test_predict))
print (classification_report(label_test, review_test_predict))
print (confusion_matrix(label_test, review_test_predict))

# Train All
r2v = Review2Vector()
review_count = r2v.fit_transform(review)
test_count = r2v.transform(test)
clf = MyNaiveBayes()
clf.fit(review_count, label)
test_predict = clf.predict(test_count)
predict_df = pd.DataFrame({'id': test_set['id'], 'sentiment': test_predict})
predict_df.to_csv('../output/nb_hand.csv', index = False, header = True)
# Stemming and Lemmatization
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

class StemLemmatization:
    @classmethod
    def do(cls, review_raw):
        review = review_raw.copy(deep = True)
        for i, one_review in review.iteritems():
            clean = re.sub("[^a-zA-Z]", " ", one_review)
            clean = clean.lower().split()
            ps = PorterStemmer()
            clean = list(map(ps.stem, clean))
            lemm = WordNetLemmatizer()
            clean = list(map(lemm.lemmatize, clean))
            review[i] = " ".join(clean)
        return review

review_train_stemlemm = StemLemmatization().do(review_train)
review_test_stemlemm = StemLemmatization().do(review_test)

# Try TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 3), use_idf=1, smooth_idf=1, stop_words = 'english')
review_train_count = tfidf.fit_transform(review_train_stemlemm)
review_test_count = tfidf.transform(review_test_stemlemm)
print(review_train_count.toarray())
# Try StemLemm+TF-IDF+MultinomialNB
clf = MultinomialNB()
clf.fit(review_train_count, label_train)
review_test_predict = clf.predict(review_test_count)
print(accuracy_score(label_test, review_test_predict))

# Get Highest Score with This Method
# Train All and Output
review_stemlemm = StemLemmatization().do(review)
test_stemlemm = StemLemmatization().do(test)
print ('Stem And Lemm Done')
tfidf = TfidfVectorizer(ngram_range=(1, 3), use_idf=1, smooth_idf=1, stop_words = 'english')
review_count = tfidf.fit_transform(review_stemlemm)
test_count = tfidf.transform(test_stemlemm)
print ('Vectorizing Done')
clf = MultinomialNB()
clf.fit(review_count, label)
test_predict = clf.predict(test_count)
print ('Predict Done')
predict_df = pd.DataFrame({'id': test_set['id'], 'sentiment': test_predict})
predict_df.to_csv('../output/nb_stemlemm_tfidf.csv', index = False, header = True)
# Try StemLemm+TF-IDF+BernoulliNB
from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()
clf.fit(review_train_count, label_train)
review_test_predict = clf.predict(review_test_count)
print(accuracy_score(label_test, review_test_predict))
# Try StemLemm+TF-IDF+GaussianNB
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.partial_fit(review_train_count.toarray(), label_train)
review_test_predict = clf.predict(review_test_count)
print(accuracy_score(label_test, review_test_predict))