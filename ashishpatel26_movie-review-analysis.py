import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC,NuSVC,SVC
from scipy import sparse
from wordcloud import WordCloud
import warnings
%matplotlib inline
pd.set_option('display.max_colwidth', -1)
warnings.filterwarnings('ignore')


from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
def prepare_text(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('[%s]' % re.escape(string.digits), '', text)
    text = re.sub('[%s]' % re.escape(' +'), ' ', text)
    text = text.lower()
    text = text.strip()
    return text

def train_predict(clf,test_data,train_feature_vector,test_feature_vector,label):
    clf.fit(train_feature_vector,label)
    prediction = clf.predict(test_feature_vector)
    test_data['Sentiment'] = prediction
    submission = test_data[['PhraseId','Sentiment']]
    return submission,prediction

def stem_input(row):
    stemmer = PorterStemmer()
    row['stemmed_phrase'] = ' '.join([stemmer.stem(word.strip()).encode('utf-8') for word in row['Phrase'].split(' ')])
    return row
train = pd.read_table('../input/movie-review-sentiment-analysis-kernels-only/train.tsv',delimiter="\t",encoding="utf-8")
test = pd.read_table('../input/movie-review-sentiment-analysis-kernels-only/test.tsv',delimiter="\t",encoding="utf-8")
train.isna().sum()
train.isnull().sum()
train['sentiment_label'] = ''
train.loc[train.Sentiment == 0, 'sentiment_label'] = 'Negative'
train.loc[train.Sentiment == 1, 'sentiment_label'] = 'Somewhat Negative'
train.loc[train.Sentiment == 2, 'sentiment_label'] = 'Neutral'
train.loc[train.Sentiment == 3, 'sentiment_label'] = 'Somewhat Positive'
train.loc[train.Sentiment == 4, 'sentiment_label'] = 'Positive'
train.head()
train.sentiment_label.value_counts()
train.shape
fig, ax = plt.subplots(1, 1,dpi=100, figsize=(10,5))
sentiment_labels = train.sentiment_label.value_counts().index
sentiment_count = train.sentiment_label.value_counts()
sns.barplot(x=sentiment_labels,y=sentiment_count)
ax.set_ylabel('Count')    
ax.set_xlabel('Sentiment Label')
ax.set_xticklabels(sentiment_labels , rotation=30)
train['cleaned_phrase'] = ''
train['cleaned_phrase'] = [prepare_text(phrase) for phrase in train.Phrase]
test['cleaned_phrase'] = ''
test['cleaned_phrase'] = [prepare_text(phrase) for phrase in test.Phrase]
# train['stemmed_phrase'] = ''
# train = train.apply(stem_input,axis=1)
# test['stemmed_phrase'] = ''
# test = test.apply(stem_input,axis=1)
train['phrase_length'] = [len(sent.split(' ')) for sent in train.cleaned_phrase]
test['phrase_length'] = [len(sent.split(' ')) for sent in test.cleaned_phrase]
train_phrase_length = sparse.csr_matrix(train.phrase_length)
test_phrase_length = sparse.csr_matrix(test.phrase_length)
train.head()
test.head()
Stopwords = list(ENGLISH_STOP_WORDS) + stopwords.words()
def wordcloud(sentiment):
    stopwordslist = Stopwords
    ## extend list of stopwords with the common words between the 3 classes which is not helpful to represent them
    stopwordslist.extend(['movie','movies','film','nt','rrb','lrb','make','work','like','story','time','little'])
    reviews = train.loc[train.Sentiment.isin(sentiment)]
    print("Word Cloud for Sentiment Labels: ", reviews.sentiment_label.unique())
    phrases = ' '.join(reviews.cleaned_phrase)
    words = " ".join([word for word in phrases.split()])
    wordcloud = WordCloud(stopwords=stopwordslist,width=3000,height=2500,background_color='white',).generate(words)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

wordcloud([3,4])
wordcloud([0,1])
wordcloud([2])
tfidf_vectorizer = TfidfVectorizer(analyzer=u'word',stop_words=Stopwords,ngram_range=(1,3), max_df = 0.5, min_df = 5)
train_tf_feature_vector = tfidf_vectorizer.fit_transform(train.cleaned_phrase)
test_tf_feature_vector = tfidf_vectorizer.transform(test.cleaned_phrase)
tfidf_char_vectorizer = TfidfVectorizer(analyzer=u'char',stop_words=Stopwords,ngram_range=(2,6), max_df = 0.5, min_df = 5)
train_tf_char_feature_vector = tfidf_char_vectorizer.fit_transform(train.cleaned_phrase)
test_tf_char_feature_vector = tfidf_char_vectorizer.transform(test.cleaned_phrase)
tf_train = sparse.hstack([train_tf_feature_vector,train_tf_char_feature_vector,train_phrase_length.T])
tf_test = sparse.hstack([test_tf_feature_vector,test_tf_char_feature_vector,test_phrase_length.T])
train.shape,tf_train.shape,test.shape,tf_test.shape
from sklearn.metrics import accuracy_score
SVM_model = LinearSVC()
svm_submission,prediction = train_predict(SVM_model,test,tf_train,tf_test,train.Sentiment.values)
svm_submission.to_csv('submission.csv',encoding="utf-8",index=None)
svm_submission.to_csv('submission.csv',encoding="utf-8",index=None)
print("Accuracy Score: ",accuracy_score(tf_test,predictiion))
