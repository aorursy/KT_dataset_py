import numpy as np # linear algebra
import pandas as pd

true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
false = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')


true['target'] = 1
false['target'] = 0

# lets combine our Dataset

data = pd.concat([true, false], axis= 0)
data.head()
# a
data['combine'] = data['title'] + data['text'] + data['subject']
data['combine'] = data['combine'].astype('str')


train = data['combine'].astype('str')
target = data.target
target
def preprocess(text):
    processed_text =text.str.replace(r'\d+(\.\d+)?', 'numbr')
    # Remove punctuation
    processed_text = processed_text.str.replace(r'[^\w\d\s]', ' ')
    
    # Replace whitespace between terms with a single space
    processed_text = processed_text.str.replace(r'\s+', ' ')
    
    # Remove leading and trailing whitespace
    processed_text = processed_text.str.replace(r'^\s+|\s+?$', '')
    
    # change words to lower case - Hello, HELLO, hello are all the same word
    processed_text = processed_text.str.lower()
    
    return processed_text
a= 'the is very good  faltu picture !@# $this getting'
a
result = preprocess(train)
result
from sklearn.feature_extraction.text import TfidfVectorizer
word_vectorizer=TfidfVectorizer(min_df=1, smooth_idf=True, norm="l2",tokenizer=lambda x: x.split(),sublinear_tf=True, ngram_range=(1,3))    
word_vectorizer.fit(result)
tfidf_trans = word_vectorizer.transform(result)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_trans , target, test_size=0.33, random_state=42)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
np.mean(predicted == y_test)
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier().fit(X_train, y_train)
predicted = clf.predict(X_test)
np.mean(predicted == y_test)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
clf=OneVsRestClassifier(LinearSVC(penalty="l2",loss='hinge',class_weight = "balanced"), n_jobs=-1)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
np.mean(predicted == y_test)