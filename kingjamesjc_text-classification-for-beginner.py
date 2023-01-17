import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv")
test= pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
# Preview
train.head()
test.head()
sample_submission.head()
train.info()
test.info()
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
stopwords_lst = stopwords.words('english')
lmtzr = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def text_process(sentence):
    """
    Takes in a string of text, then performs the following:
    1. Stemming
    2. Remove all punctuation
    3. Remove all stopwords
    4. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in sentence if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc).lower()
    # Now just remove any stopwords
    return [stemmer.stem(word) for word in nopunc.split() if word not in stopwords_lst]
x_train, x_test, y_train, y_test, idx1,idx2 = train_test_split(train["text"], train["author"],train["id"], random_state=42)
mnb_pipeline = Pipeline([('bow', TfidfVectorizer(analyzer=text_process, ngram_range=(1,2))),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', MultinomialNB())
])
mnb_pipeline.fit(x_train,y_train)
mnb_predictions = mnb_pipeline.predict(x_test)
print(classification_report(mnb_predictions, y_test))
from sklearn.linear_model import SGDClassifier
svm_pipeline = Pipeline([('bow', TfidfVectorizer(analyzer=text_process, ngram_range=(1,2), max_df=0.8)),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', SGDClassifier(loss='hinge', penalty='l2', 
                                                 alpha=1e-3, random_state=42, 
                                                 max_iter=5, tol=None))
])
svm_pipeline.fit(x_train,y_train)
svm_predictions = svm_pipeline.predict(x_test)
print(classification_report(svm_predictions, y_test))
X_train = train["text"].as_matrix().T
y_train = train["author"].as_matrix().T
mnb_pipeline.fit(X_train, y_train)
X_test = test["text"].as_matrix().T
prediction = mnb_pipeline.predict_proba(X_test)
submission_df = pd.DataFrame(prediction)
submission_df = submission_df.join(test)
col=["EAP","HPL","MWS"]
submission_df.rename(columns={0:"EAP",1:"HPL",2:"MWS"}, inplace=True)
submission_df = submission_df[["id","EAP","HPL","MWS"]]
submission_df.to_csv("submission.csv",header=True)
submission_df.head()
