import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fast')
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from sklearn.metrics import accuracy_score
stop_words = set(stopwords.words('english'))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import itertools
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from wordcloud import WordCloud, STOPWORDS
from sklearn.multiclass import OneVsRestClassifier
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
data = pd.read_csv('../input/Tweets.csv')
data.head()
print(data.shape)
print(data.airline_sentiment.unique())
data.isnull().sum()/data.shape[0]
Index=[1, 2, 3]
print(data.airline_sentiment.value_counts())
plt.bar(Index, data.airline_sentiment.value_counts())
plt.xticks(Index, ['negative', 'neutral', 'positive'], rotation = 45)
plt.ylabel('Number of tweets')
plt.xlabel('Sentiment expressed in tweets')
print(data.airline_sentiment.value_counts() / data.shape[0])
df=data.groupby(["airline","airline_sentiment"]).size().unstack()
print(df)
ax=df.plot.bar(stacked=True)
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
df=data.groupby(["airline","airline_sentiment"]).size().unstack()
df=df.div(df.sum(axis=1),axis='index')
print(df)
ax=df.plot.bar(stacked=True)
plt.ylabel('Fraction of Tweets')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
negative_tweets=data[(data.airline_sentiment=="negative") & (data.negativereason !="Can't Tell")]
df=negative_tweets.groupby(["negativereason"]).size().sort_values()
df.plot.bar()
plt.ylabel('Number of Tweets')
df=negative_tweets.groupby(["airline","negativereason"]).size().unstack()
ax=df.plot.bar(stacked=True)
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
df=negative_tweets.groupby(["airline","negativereason"]).size().unstack()
df=df.div(df.sum(axis=1),axis='index')#rowsum
ax=df.plot.bar(stacked=True)
plt.ylabel('Fraction of Tweets')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data_bak = data.copy
data.head()
# apostrophe lookup dict
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
lemmatizer = WordNetLemmatizer()
import emoji

test_str = "Thank you @VirginAmerica for you amazing 👍 customer support team on Tuesday 11/28 at @EWRairport and returning my lost bag in less than 24h! #efficiencyiskey #virginamerica"

def clean_text(text):
    text = re.sub(r'(?:@[\w_]+)', " ", text) # @-mentions
    text = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", " ", text) # hash-tags
    text = emoji.demojize(text).replace('_','')
    #text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w{1,1}\b', '', text)
    #text = re.sub(r'http.?://[^\s]+[\s]?', '', text) #Remove URLs
    return text

test_str = clean_text(test_str)

def cleanupText(s):
    stopset = set(stopwords.words('english'))
    stopset.add('wikipedia')

    tokens =sequence=text_to_word_sequence(s, 
                                        filters="\"!'#$%&()*+,-˚˙./:;‘“<=·>?@[]^_`{|}~\t\n",
                                        lower=True,
                                        split=" ")
    tokens=[APPO[token] if token in APPO else token for token in tokens]
    for token in tokens:
        lemmatizer.lemmatize(token, 'v')
    cleanup = " ".join(filter(lambda word: word not in stopset, tokens))
    return cleanup

print(cleanupText(test_str))
data.text = data.text.map(lambda text : clean_text(text))
data.text = data.text.apply(cleanupText)
df = data[data['airline_sentiment']=='negative']
words = ' '.join(df['text'])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(words)
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
df = data[data['airline_sentiment']=='positive']
words = ' '.join(df['text'])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(words)
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

lb = LabelEncoder()
data['sentiment_encoded'] = lb.fit_transform(data['airline_sentiment'])
data[['airline_sentiment', 'sentiment_encoded']]
count_vect = CountVectorizer(decode_error='ignore',stop_words='english')
count_vect.fit_transform(data.text)
print(data.airline.unique())
X = data.text
print(X.shape)
y = data['sentiment_encoded']
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20, shuffle=True)
X_train_count = count_vect.transform(X_train)
X_test_count = count_vect.transform(X_test)
print(X_train_count.shape)
print(X_test_count.shape)
y[:5]
classifier = MultinomialNB()
classifier.fit(X_train_count, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_count)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
classifier = LogisticRegression(multi_class='multinomial', random_state = 0, n_jobs = -1, solver = 'sag', C=1, max_iter = 2000)
classifier.fit(X_train_count, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_count)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=None,
    smooth_idf = True
)
word_vectorizer.fit_transform(data.text)
X_train_word = word_vectorizer.transform(X_train)
X_test_word = word_vectorizer.transform(X_test)
classifier = MultinomialNB()
classifier.fit(X_train_word, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_word)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
classifier = LogisticRegression(multi_class='multinomial', random_state = 0, n_jobs = -1, solver = 'sag', C=1)
classifier.fit(X_train_word, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_word)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
count_vect_ngram = CountVectorizer(decode_error='ignore',stop_words='english', ngram_range=(1, 2), token_pattern=r'\w{1,}')
count_vect_ngram.fit_transform(data.text)
X_train_ngram = count_vect_ngram.transform(X_train)
X_test_ngram = count_vect_ngram.transform(X_test)
classifier = MultinomialNB()
classifier.fit(X_train_ngram, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_ngram)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
classifier = LogisticRegression(multi_class='multinomial', random_state = 0, n_jobs = -1, solver = 'sag', max_iter=2000)
classifier.fit(X_train_ngram, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_ngram)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
new_features = pd.get_dummies(data['airline'],prefix='airline')
X = data.text
print(data.shape)
print(new_features.shape)
print(type(X))
print(type(new_features))
X = pd.concat([X, new_features], axis = 1)
#print(X)
print(X.shape)
y = data['sentiment_encoded']
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20, shuffle=True)
X_train_ngram = count_vect_ngram.transform(X_train['text'])
X_test_ngram = count_vect_ngram.transform(X_test['text'])
X_train_ngram = hstack([X_train_ngram, X_train.iloc[:, 1:]])
X_test_ngram = hstack([X_test_ngram, X_test.iloc[:, 1:]])
print(X_train_ngram.shape)
print(X_test_ngram.shape)
classifier = MultinomialNB()
classifier.fit(X_train_ngram, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_ngram)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
classifier = OneVsRestClassifier(SVC(random_state = 0, kernel='linear',gamma=0.01, C = 0.1, probability=True))
classifier.fit(X_train_ngram, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_ngram)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
classifier = OneVsRestClassifier(LogisticRegression(random_state = 0, n_jobs = -1, solver = 'sag', C=1, max_iter= 2000))
classifier.fit(X_train_ngram, y_train)
final_classifier_sentiment = classifier

# Predicting the Test set results
y_pred = classifier.predict(X_test_ngram)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
data[data['sentiment_encoded'] == 0].head()
negative_data = data[data['sentiment_encoded'] == 0]
cols = ['negativereason', 'airline', 'text'] 
negative_data[cols].head()
negative_data.negativereason.value_counts()
negative_data.negativereason.value_counts() / negative_data.shape[0]
def fix_reason(text):
    if text not in ['Customer Service Issue', 'Late Flight']:
        text = 'Others'
    return text

negative_data['negativereason_fixed'] = negative_data.negativereason.map(lambda text : fix_reason(text))
cols = ['negativereason', 'negativereason_fixed', 'airline', 'text']
negative_data[cols].tail()
negative_data.negativereason_fixed.value_counts()
negative_data.negativereason_fixed.value_counts() / negative_data.shape[0]
negative_data['negativereason_fixed_encoded'] = lb.fit_transform(negative_data['negativereason_fixed'])
negative_data[['negativereason_fixed', 'negativereason_fixed_encoded']]
new_features = pd.get_dummies(negative_data['airline'],prefix='airline')
X = negative_data.text
print(negative_data.shape)
print(new_features.shape)
print(type(X))
print(type(new_features))
X = pd.concat([X, new_features], axis = 1)
#print(X)
print(X.shape)
y = negative_data['negativereason_fixed_encoded']
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20, shuffle=True)
X_train_ngram = count_vect_ngram.transform(X_train['text'])
X_test_ngram = count_vect_ngram.transform(X_test['text'])
X_train_ngram = hstack([X_train_ngram, X_train.iloc[:, 1:]])
X_test_ngram = hstack([X_test_ngram, X_test.iloc[:, 1:]])
print(X_train_ngram.shape)
print(X_test_ngram.shape)
classifier = MultinomialNB()
classifier.fit(X_train_ngram, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_ngram)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
classifier = OneVsRestClassifier(SVC(random_state = 0, kernel='linear',gamma=0.01, C = 0.1, probability=True))
classifier.fit(X_train_ngram, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_ngram)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
classifier = OneVsRestClassifier(LogisticRegression(random_state = 0, n_jobs = -1, solver = 'sag', C=1, max_iter= 2000))
classifier.fit(X_train_ngram, y_train)
final_classifier_negativereason = classifier

# Predicting the Test set results
y_pred = classifier.predict(X_test_ngram)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))