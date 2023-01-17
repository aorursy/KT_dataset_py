import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

import spacy
from spacy import displacy
nlp = spacy.load('en')

import nltk
from nltk.corpus import treebank
nltk.download('treebank')

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
pd.options.mode.chained_assignment = None

# for Sentiment Analysis
from textblob import TextBlob
filename = '../input/export-dashboard-tsla/export_dashboard_tsla.xlsx'
tslaTweets = pd.read_excel(filename,sheet_name='Stream')
tslaTweets
# pre-processing fuction

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)
# call the normalize_document function to pre-process the training corpus

norm_content = normalize_corpus(tslaTweets['Tweet content'])
norm_content
tslaTweets.shape
norm_content.shape
tslaTweets['cleaned'] = norm_content
tslaTweets.head()
# Filter the contents 
LANG = 'en'
HASH_TAGS = '#Tesla'

# Filter with the langauge 
tslaTweetsFiltered = tslaTweets[(tslaTweets['Tweet language (ISO 639-1)'] == LANG)]

# Filter with the langauge and hastags (note: use contains for hashtags)
#tslaTweetsFiltered = tslaTweets[(tslaTweets['Tweet language (ISO 639-1)'] == LANG) & (tslaTweets['Hashtags'].str.contains(HASH_TAGS))]

tslaTweetsFiltered.reset_index(drop=True, inplace=True)
tslaTweetsFiltered.head()
tslaTweetsFiltered.columns
# Look at the subset of useful columns for the sentiment trading
COLUMNS = ['Date', 'cleaned', 'Followers']
tslaTweetsSubset = tslaTweetsFiltered[COLUMNS]

# Convert Date string to datetime to match with the stock daily change later
tslaTweetsSubset['Date'] = pd.to_datetime(tslaTweetsSubset['Date'])
tslaTweetsSubset.head()
tslaTweetsSubset.shape
tslaTweetsSubset = tslaTweetsSubset.dropna()
tslaTweetsSubset
# Use TextBlob to run the tweets sentiment polarity
tslaTweetsSubset['sentiment'] = tslaTweetsSubset['cleaned'].apply(lambda x: TextBlob(x).polarity)

# Weight the tweets sentiment importance by the number of followings
tslaTweetsSubset['sentiment_weighted'] = tslaTweetsSubset['sentiment'] * tslaTweetsSubset['Followers']
tslaTweetsSubset
# Check Sentiment assessment
INDEX = 0

text = tslaTweetsSubset.iloc[INDEX]['cleaned']
print(text)

TextBlob(text).sentiment_assessments
# Group the weighted sentiment by Date for matching the stock daily change
aggregateSentiments = tslaTweetsSubset.groupby(['Date']).sum()[['sentiment_weighted']]
aggregateSentiments
# get stocks daily data (OHLCV) from Yahoo
import pandas_datareader.data as web
from datetime import datetime

start = datetime(2016, 4, 2) 
end = datetime(2016, 6, 15) 
stock= web.DataReader('TSLA', 'yahoo', start=start, end=end)
stock.head()
# calculate the stock daily change
stock['change'] = (stock['Close'] - stock['Open']) / stock['Open']
stock.head()
start = datetime(2016, 4, 2) 
end = datetime(2016, 6, 15) 
NDX= web.DataReader('^NDX', 'yahoo', start=start, end=end)
NDX.head()
# calculate the NASDAQ 100 daily change
NDX['NDX change'] = (NDX['Close'] - NDX['Open']) / NDX['Open']
NDX.head()
# Merge the daily stock price info with the sentiments
merged = stock.merge(aggregateSentiments, on='Date', how='left')[['change', 'sentiment_weighted']]
merged.head()
# Scale the unit to -1 to 1
scaler = MinMaxScaler((-1, 1))
merged['changes'] = scaler.fit_transform(merged[['change']])
merged['sentiments'] = scaler.fit_transform(merged[['sentiment_weighted']])
scaled = merged[['changes', 'sentiments']]
scaled = scaled.dropna()
scaled.head()
scaled.plot(figsize=(10, 6))
# shows the correlation
corr_data = scaled.corr()
corr_data.style.background_gradient(cmap='coolwarm', axis=None)
# Try sentiments with different date lags

# Sentiment shift backwards -> Current day sentiments predicts next day stock price change (predictive)
scaled['sentiment-1'] = merged['sentiments'].shift(-1)

# Sentiment shift forwards -> Current day sentiments reflects yesterday's price change (reactive)
scaled['sentiment+1'] = merged['sentiments'].shift(1)
scaled.head()
scaled.plot(figsize=(10, 6))
corr_data = scaled.corr()
corr_data.style.background_gradient(cmap='coolwarm', axis=None)
stock.head()
# Look at the subset of useful columns for the NASDAQ 100
COLUMNS = ['Open', 'Close', 'NDX change']
NDX = NDX[COLUMNS]
NDX.head()
# Look at the subset of useful columns for the Tesla Stock
COLUMNS = ['Open', 'Close', 'change']
stock = stock[COLUMNS]
stock.head()
#combine the information of NASDAQ 100 and Tesla
data_df = pd.merge(stock,NDX,on='Date',how='left')
data_df.head()
#calculate the difference of between Tesla change and NASDAQ 100 change
data_df['difference'] = data_df['change'] - data_df['NDX change']
data_df.head()
#Add tag to all dates, if the tag is 1 tesla stock return is higher than NASDAQ 100 on that day otherwise the tag is 0

data_df['tag']=[1 if x>0 else 0 for x in data_df['difference'] ]
data_df.head()
data_df = pd.merge(data_df,tslaTweetsSubset,on='Date',how='left')
data_df.head()
from sklearn.model_selection import train_test_split 

X = np.array(data_df['cleaned'].fillna(' ')) 
y = np.array(data_df['tag'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer
   
### build BOW features on train_corpus using the CountVectortizer

### transform the test_corpus using the transform method

transfer = CountVectorizer()

X_train = transfer.fit_transform(X_train)

X_test = transfer.transform(X_test)
# Naïve Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

# fit the model with the y_train as the target and the BOW featurs on the train_corpus as input

estimator = MultinomialNB()

estimator.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix
# Predict the testing output using the BOW features on the test_corpus

#compare the real value with predicted value
y_predict = estimator.predict(X_test)

labels = np.unique(y_test)

cm = confusion_matrix(y_test, y_predict, labels=labels)

pd.DataFrame(cm, index=labels, columns=labels)
# calculate the accuracy, precison, recall and F1 - score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_predict, y_test)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_predict, y_test)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_predict, y_test)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_predict, y_test)
print('F1 score: %f' % f1)
# Compute ROC curve and AUC for the predicted class "1"

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

y_score = estimator.predict_proba(X_test)[:,1]

# Compute ROC curve and AUC for the predicted class "1"

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, threshold = roc_curve(y_test, y_score)

# Compute Area Under the Curve (AUC) using the trapezoidal rule
estimator_roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
print("AUC score = {}".format(estimator_roc_auc))
# verify

fpr, tpr, threshold = roc_curve(y_test, y_score)
estimator_roc_auc = auc(fpr, tpr)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.set_title('Receiver Operating Characteristic')
ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % estimator_roc_auc)
ax.legend(loc = 'lower right')
ax.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')
from sklearn.model_selection import train_test_split 

X = np.array(data_df['cleaned'].fillna(' ')) 
y = np.array(data_df['tag'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
### build BOW features on train_corpus using the TfidfVectortizer
### transform the test_corpus using the transform method

transfer = TfidfVectorizer()

X_train = transfer.fit_transform(X_train)

X_test = transfer.transform(X_test)
# Naïve Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

# fit the model with the y_train as the target and the BOW featurs on the train_corpus as input

estimator = MultinomialNB()

estimator.fit(X_train,y_train)
# Predict the testing output using the BOW features on the test_corpus

#compare the real value with predicted value
y_predict = estimator.predict(X_test)
print("y_predict:",y_predict)
print("The boolen matrix of prediction:",y_test==y_predict)   #True means the model predicting correctly on that article!

#precison
score = estimator.score(X_test,y_test)
# Evaluate the model using the confusion matrix
import pandas as pd
from sklearn.metrics import confusion_matrix

labels = np.unique(y_test)
cm = confusion_matrix(y_test, y_predict, labels=labels)

pd.DataFrame(cm, index=labels, columns=labels)
# calculate the accuracy, precison, recall and F1 - score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_predict, y_test)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_predict, y_test)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_predict, y_test)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_predict, y_test)
print('F1 score: %f' % f1)
# Compute ROC curve and AUC for the predicted class "1"

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

y_score = estimator.predict_proba(X_test)[:,1]

# Compute ROC curve and AUC for the predicted class "1"

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, threshold = roc_curve(y_test, y_score)

# Compute Area Under the Curve (AUC) using the trapezoidal rule
estimator_roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
print("AUC score = {}".format(estimator_roc_auc))
# verify

fpr, tpr, threshold = roc_curve(y_test, y_score)
estimator_roc_auc_roc_auc = auc(fpr, tpr)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.set_title('Receiver Operating Characteristic')
ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % estimator_roc_auc_roc_auc)
ax.legend(loc = 'lower right')
ax.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')