import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize.toktok import ToktokTokenizer
from snowballstemmer import TurkishStemmer
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.naive_bayes import MultinomialNB
import re

pd.options.display.max_colwidth = 280 # For showing big tweets
%matplotlib inline
import os
import warnings
warnings.filterwarnings('ignore')

print(sklearn.__version__)
print(matplotlib.__version__)
print(np.__version__)
print(pd.__version__)
print(nltk.__version__)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

turkcell_data = pd.read_csv("../input/twitter-turkcell-data/turkcell.csv")
turkcell_data = turkcell_data.drop(['created_at', 'date', 'time', 'username'], axis = 1)
turkcell_data.columns = ['Tweets', 'Sentiment']
turkcell_data.Sentiment.replace([0, 4], ['Negative', 'Positive'], inplace = True)
turkcell_data.head()
turkcell_data.describe()
turkcell_data['Sentiment'].value_counts()
turkcell_data['TextSize'] = [len(t) for t in turkcell_data.Tweets]
turkcell_data.head()
turkcell_data['TextSize'].mean()
plt.boxplot(turkcell_data.TextSize) # plot TextSize column
plt.show()
target_tdS = Counter(turkcell_data.Sentiment)

plt.figure(figsize=(16,8))
plt.bar(target_tdS.keys(), target_tdS.values())
plt.title("Dataset labels distribuition")
tweets_text = turkcell_data.Tweets.str.cat()
emoticons = set(re.findall(r" ([xX:;][-']?.) ", tweets_text))
emoticons_count = []
for emot in emoticons:
    emoticons_count.append((tweets_text.count(emot), emot))
sorted(emoticons_count, reverse=True)
happy_emot = r" ([xX;:]-?[dD)]|:-?[\)]|[;:]|[:')][pP]) "
sad_emot = r" (:'?[/|\(]) "
print("Happy emoticons:", set(re.findall(happy_emot, tweets_text)))
print("Sad emoticons:", set(re.findall(sad_emot, tweets_text)))
#Tokenization of text
tokenizer = ToktokTokenizer()

#Setting Turkish stopwords
stopword_list = open('../input/turkce-stop-words/turkce-stop-words', 'r').read().split()
def most_used_words(text):
    tokens = tokenizer.tokenize(text)
    frequency_dist = nltk.FreqDist(tokens)
    print("There is %d different words" % len(set(tokens)))
    return sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)
sorted(most_used_words(turkcell_data.Tweets.str.cat())[:150])
mw = most_used_words(turkcell_data.Tweets.str.cat())
most_words = []
for w in mw:
    if len(most_words) == 1000:
        break
    if w in stopword_list:
        continue
    else:
        most_words.append(w)
sorted(most_words[:150])
# Remove the html
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Remove the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Remove the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

turkcell_data['Tweets'] = turkcell_data['Tweets'].apply(denoise_text)
def stemWord(text):
    return text.lower()

# Remove usernames
def removeUsernames(text):
    return re.sub('@[^\s]+', '', text)

# Remove hashtags
def removeHashtags(text):
    return re.sub(r'#[^\s]+', ' ', text)

# Remove punctuation
def removePunctuation(text):
    return re.sub(r'[^\w\s]', ' ', text)

# Remove single character
def singleCharacterRemove(text):
    return re.sub(r'(?:^| )\w(?:$| )', ' ', text)
                  
# Remove emoticon
def stripEmoji(text):
    emoji = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    return emoji.sub(r'', text)

def splitIntoStem(text):
    text = stemWord(text)
    text = removeUsernames(text)
    text = removeHashtags(text)
    text = removePunctuation(text)
    text = singleCharacterRemove(text)
    text = stripEmoji(text)
    return text


turkcell_data['Tweets'] = turkcell_data['Tweets'].apply(splitIntoStem)
# Remove pic.twitter
def picTwitter(text):
    pic_pat = r'pic.[^ ]+'
    text = re.sub(pic_pat, '', text)
    return text
    
turkcell_data['Tweets'] = turkcell_data['Tweets'].apply(picTwitter)

# Remove pic.twitter2
def picTwitter2(text):
    pic_pat2 = r'com.[^ ]+'
    text = re.sub(pic_pat2, '', text)
    return text

turkcell_data['Tweets'] = turkcell_data['Tweets'].apply(picTwitter2)
'''# Remove turkcell
def operator(text):
    t = r'turkcell'
    v = r'vodafone'
    tt = r'turktelekom'
    so = r'superonline'
    sy = r'superyardim'
    text = re.sub(t, '', text)
    text = re.sub(v, '', text)
    text = re.sub(tt, '', text)
    return text


turkcell_data['Tweets'] = turkcell_data['Tweets'].apply(operator)'''
turkcell_data['Tweets']
turkcell_data.head()
turkcell_data['TextSizeBeforeRemoveStopWords'] = [len(t) for t in turkcell_data.Tweets]
turkcell_data.head()
# Set Stopwords to Turkish
stop = set(stopword_list)

def remove_stopwords(text, is_lower_case = True):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

turkcell_data['Tweets'] = turkcell_data['Tweets'].apply(remove_stopwords)
turkcell_data['TextSizeAfterRemoveStopWords'] = [len(t) for t in turkcell_data.Tweets]
turkcell_data.head()
turkStem = TurkishStemmer()

# Stemming
def simple_stemmer(text):
    ss = TurkishStemmer()
    text = ' '.join([ss.stemWord(word) for word in text.split()])
    return text

turkcell_data['Tweets'] = turkcell_data['Tweets'].apply(simple_stemmer)
turkcell_data['TextSizeStemming'] = [len(t) for t in turkcell_data.Tweets]
turkcell_data.head()
turkcell_data.Sentiment.replace(['Negative', 'Positive'], [0, 1], inplace = True)
turkcell_data.head()
sent = turkcell_data['Sentiment']
tw = turkcell_data['Tweets']

# Split the data / Ratio is 80:20
X_train, X_test, y_train, y_test = train_test_split(tw, sent, test_size = 0.20, random_state= 1)


# X_train is the tweets of training data, 
# X_test is the testing tweets which we have to predict 
# y_train is the sentiments of tweets in the traing data
# y_test is the sentiments of the tweets which we will use to measure the accuracy of the model
# CountVectorizer for Bag of Words
cv = CountVectorizer(min_df = 0, max_df = 1, binary = False, ngram_range = (1, 3))

# Transformed train tweets
cv_train_tweets = cv.fit_transform(X_train)

# Transformed test tweets
cv_test_tweets = cv.transform(X_test)

print('BoW_CV_Train:',cv_train_tweets.shape)
print('BoW_CV_Test:',cv_test_tweets.shape)
s = cv_train_tweets[1]
print(s)
ss = cv_test_tweets[1]
print(ss)
# TfidfVectorizer
tv = TfidfVectorizer(min_df = 0, max_df = 1, use_idf = True, ngram_range = (1, 3))

# Transformed train tweets
tv_train_tweets = tv.fit_transform(X_train)

# Transformed test tweets
tv_test_tweets = tv.transform(X_test)

print('Tfidf_Train:',tv_train_tweets.shape)
print('Tfidf_Test:',tv_test_tweets.shape)
s = tv_train_tweets[1]
print(s)
ss = tv_test_tweets[1]
print(ss)
# Training the Model
lr = LogisticRegression(penalty = 'l2', max_iter = 500, C = 1.1, random_state = 42)

# Fitting the model for Bag of Words
lr_bow = lr.fit(cv_train_tweets, y_train)
print(lr_bow)

# Fitting the model for TFIDF features
lr_tfidf = lr.fit(tv_train_tweets, y_train)
print(lr_tfidf)
# Predicting the model for Bag of Words
lr_bow_predict = lr.predict(cv_test_tweets)
print(lr_bow_predict)

# Predicting the model for TFIDF features
lr_tfidf_predict = lr.predict(tv_test_tweets)
print(lr_tfidf_predict)
# Accuracy score for Bag of Words
lr_bow_score = accuracy_score(y_test, lr_bow_predict)
print("LR BoW Score :",lr_bow_score)

# Accuracy score for TFIDF features
lr_tfidf_score = accuracy_score(y_test, lr_tfidf_predict)
print("LR TFIDF Score :",lr_tfidf_score)
# Classification report for Bag of Words
lr_bow_report = classification_report(y_test, lr_bow_predict, target_names = ['Positive','Negative'])
print(lr_bow_report)

# Classification report for TFIDF features
lr_tfidf_report = classification_report(y_test, lr_tfidf_predict, target_names = ['Positive','Negative'])
print(lr_tfidf_report)
# Confusion matrix for Bag of Words
cm_bow = confusion_matrix(y_test, lr_bow_predict, labels = [1,0])
print(cm_bow)

# Confusion matrix for TFIDF features
cm_tfidf = confusion_matrix(y_test, lr_tfidf_predict, labels = [1,0])
print(cm_tfidf)
# Training the Linear SVM
svm = SGDClassifier(loss='hinge', max_iter=500, random_state=42)

# Fitting the SVM for Bag of Words
svm_bow = svm.fit(cv_train_tweets, y_train)
print(svm_bow)

# Fitting the SVM for TFIDF features
svm_tfidf = svm.fit(tv_train_tweets, y_train)
print(svm_tfidf)
# Predicting the model for Bag of Words
svm_bow_predict = svm.predict(cv_test_tweets)
print(svm_bow_predict)

# Predicting the model for TFIDF features
svm_tfidf_predict = svm.predict(tv_test_tweets)
print(svm_tfidf_predict)
# Accuracy score for Bag of Words
svm_bow_score = accuracy_score(y_test, svm_bow_predict)
print("SVM BoW Score :",svm_bow_score)

# Accuracy score for TFIDF features
svm_tfidf_score = accuracy_score(y_test, svm_tfidf_predict)
print("SVM TFIDF Score:",svm_tfidf_score)
# Classification report for Bag of Words 
svm_bow_report = classification_report(y_test, svm_bow_predict, target_names = ['Positive','Negative'])
print(svm_bow_report)

# Classification report for TFIDF features
svm_tfidf_report = classification_report(y_test, svm_tfidf_predict, target_names = ['Positive','Negative'])
print(svm_tfidf_report)
# Confusion matrix for Bag of Words
cm_bow = confusion_matrix(y_test, svm_bow_predict, labels = [1,0])
print(cm_bow)

# Confusion matrix for TFIDF features
cm_tfidf = confusion_matrix(y_test, svm_tfidf_predict, labels = [1,0])
print(cm_tfidf)
# Training the model
mnb = MultinomialNB()

# Fitting the NB for Bag of Words
mnb_bow = mnb.fit(cv_train_tweets, y_train)
print(mnb_bow)

# Fitting the NB for TFIDF features
mnb_tfidf = mnb.fit(tv_train_tweets, y_train)
print(mnb_tfidf)
# Predicting the model for Bag of Words
mnb_bow_predict = mnb.predict(cv_test_tweets)
print(mnb_bow_predict)

# Predicting the model for TFIDF features
mnb_tfidf_predict = mnb.predict(tv_test_tweets)
print(mnb_tfidf_predict)
# Accuracy score for Bag of Words
mnb_bow_score = accuracy_score(y_test, mnb_bow_predict)
print("MNB BoW Score :",mnb_bow_score)

# Accuracy score for TFIDF features
mnb_tfidf_score = accuracy_score(y_test, mnb_tfidf_predict)
print("MNB TFIDF Score :",mnb_tfidf_score)
# Classification report for Bag of Words
mnb_bow_report = classification_report(y_test, mnb_bow_predict, target_names = ['Positive','Negative'])
print(mnb_bow_report)

# Classification report for TFIDF features
mnb_tfidf_report = classification_report(y_test, mnb_tfidf_predict, target_names = ['Positive','Negative'])
print(mnb_tfidf_report)
# Confusion matrix for Bag of Words
cm_bow = confusion_matrix(y_test, mnb_bow_predict, labels = [1,0])
print(cm_bow)

# Confusion matrix for TFIDF features
cm_tfidf = confusion_matrix(y_test, mnb_tfidf_predict, labels = [1,0])
print(cm_tfidf)