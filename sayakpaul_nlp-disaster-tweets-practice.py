# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

import string

import re

import nltk

from sklearn.preprocessing import StandardScaler

from nltk.corpus import stopwords 

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter

from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import RidgeClassifier



# Import elements for multilayer percepton 

from keras import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train.head()
test.head()
print("Shape of training set : {}".format(train.shape))

print("Shape of test set : {}".format(test.shape))
### Lets check for the mssing values 

print("NO OF MISSING VALUES :");print("------Training set-------")

print("keyword : {}".format(train.keyword.isnull().sum()))

print("location : {}".format(train.location.isnull().sum()))

print("-------Test data--------")

print("keyword : {}".format(test.keyword.isnull().sum()))

print("location : {}".format(test.location.isnull().sum()))

print("");print("")

print("PROPORTION OF MISSING VALUES :");print("------Training set-------")

print("keyword : {}".format(train.keyword.isnull().sum()/train.shape[0]*100))

print("location : {}".format(train.location.isnull().sum()/train.shape[0]*100))

print("-------Test data--------")

print("keyword : {}".format(test.keyword.isnull().sum()/test.shape[0]*100))

print("location : {}".format(test.location.isnull().sum()/test.shape[0]*100))
### remove location

train.drop(columns=['location','keyword'],inplace=True)

test.drop(columns=['location','keyword'],inplace=True)
### class distribution of target###

sns.set(style='darkgrid', context='notebook')

sns.countplot(x='target', data=train).set_title('target distribution')

#plt.title('target distribution')
### lets take a quick look at some of the texts to guess what sort of cleaning needs to be done

for i in [5,12,44,22,45,67,99,122,455,78,2225,558,111,5578,546,447,944,6557,1115,447,6552,4177,700,4999,425]:

    print(train.text[i])

    print('\n')
## Lets see a random piece of text

train.text[455]
### Visualising through wordclouds 

texts = train.text.tolist()

texts_combined = ' '.join(texts)

plt.figure(figsize=(14,14))

plt.imshow(WordCloud().generate(texts_combined))

plt.axis("off")
### Visualising for positive texts

positive = train.text[train.target == 1]

positive_texts = positive.tolist()

positive_texts_combined = ' '.join(positive_texts)

plt.figure(figsize=(14,14))

plt.imshow(WordCloud().generate(positive_texts_combined))

plt.axis("off")

plt.title("positive texts")
### Visualising for negative texts

negative = train.text[train.target == 0]

negative_texts = negative.tolist()

negative_texts_combined = ' '.join(negative_texts)

plt.figure(figsize=(14,14))

plt.imshow(WordCloud().generate(negative_texts_combined))

plt.axis("off")

plt.title("Negative targets")
#text = "model love u take u time urð\x9f\x93± ð\x9f\x98\x99ð\x9f\x98\x8eð\x9f\x91\x84ð\x9f\x91 ð\x9f\x92¦ð\x9f\x92¦ð\x9f\x92¦"

#print(text)

#text = remove_irrelevant(text)

#text = remove_punc(text)

#text = remove_emoji(text)

#text = re.sub('[@#]*[@\w]*\d{1,}[@\w]*','',text)

#text = re.sub('\d+','',text)

#text = re.sub('_*',' ',text)

#text = re.sub('[[\w*]]*','',text)

#text = re.sub('[\t\n\r\f\v]+','',text)

#text = re.sub('@\w*','',text)

#text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)

#text = remove_punc(text)

#print("\n");print(text)

#print(text)
## example demonstrating decoding into ASCII characters



string_with_nonASCII = "àa string withé all the fuünny charactersß."



encoded_string = string_with_nonASCII.encode("ascii", "ignore")

decode_string = encoded_string.decode()



print(decode_string)
## example demonstrating removal of unicode encoded characters



string = "jbj\x9f gn9 blasts"

print(string)



string_unicode_removed = string.encode("ascii", "ignore").decode()

print(string_unicode_removed)
def remove_irrelevant(text):

    

    ## convert non-ASCII to ASCII characters

    text = text.encode("ascii","ignore").decode()

    ## remove emails

    text = re.sub('[\w\.-]+@[\w\.-]+','',text)

    ## remove words within brackets

    text = re.sub('\[.*?\]', '', text)

    ## remove words containing nos. & special characters in between

    text = re.sub('[@#-]*[@\w]*\d{1,}[-@\w]*','',text)

    ## remove numbers

    text = re.sub('\d+','',text)

    ## remove blankspace characters

    text = re.sub('[\t\n\r\f\v]+','',text)

    ## remove callouts (starting with @)

    text = re.sub('@\w*','',text)

    ## remove urls

    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)

    ## convert to lowercase

    text = text.lower()

    

    return text



### remove punctuations

def remove_punc(text):

    not_punc = [w for w in text if w not in string.punctuation]

    text_punc_removed = ''.join(not_punc)

    return text_punc_removed



def remove_emoji(text): ## remove emojis

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)    

 

### remove repetitive characters in a word

def rep(text):

    grp = text.group(0)

    if len(grp) > 1:

        return grp[0:2] # can change the value here on repetition

def unique_char(rep,sentence):

    convert = re.sub(r'(\w)\1+', rep, sentence) 

    return convert





### Lemmatization

def lemmatize(text):

    

    lemmatizer = WordNetLemmatizer()

    return " ".join([lemmatizer.lemmatize(i) for i in text.split()])
### example removal of irrelevant characters

import string

text = "fdf @nj23jnj-jn nkn! @ddsk kkm [sssd23_dd!]"



text = remove_irrelevant(text)

text = remove_punc(text)



print(text)
### example showing conversion of repetitive characters to 2 characters

sentence="heyyy givvve meee the address"

unique_char(rep,sentence)
### cleaning the texts

train["text"] = train["text"].apply(lambda x:remove_irrelevant(x))

train["text"] = train["text"].apply(lambda x:remove_punc(x))

train["text"] = train["text"].apply(lambda x:remove_emoji(x))

train["text"] = train["text"].apply(lambda x:unique_char(rep,x))

train["text"] = train["text"].apply(lambda x:lemmatize(x))



positive = train.text[train.target == 1]

negative = train.text[train.target == 0]



train.head()
### Visualising cleaned texts for identifying stopwords



words_combined = []

for i in range(train.shape[0]):

    for w in train.text[i].split():

        words_combined.append(w)

#words_combined

#print(len(words_combined))



dict = Counter(words_combined).most_common()

dict[:60]
## Adding more stopwords(unigram)



stopwords = stopwords.words("english")



add = [word for word,count in dict if count>200]+['ve','rs','ll','d','t','s']



keep = ['i','you','my','with','like','as','me','your','not','its','out','after','all','no','fire','we','get','new'

       ,'now','more','dont','about']



if add not in stopwords:

    stopwords_added = stopwords + add

stopwords_added  



stopwords_unigram = [w for w in stopwords_added if w not in keep]

print(stopwords_unigram)
### demo showing to generate n-grams



def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(' ')]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [' '.join(ngram) for ngram in ngrams]



generate_ngrams("ccc zxxc  asd asca  dvdd asfa",n_gram=2)## bigrams
# Bigrams



disaster_bigrams = defaultdict(int)

nondisaster_bigrams = defaultdict(int)



for tweet in positive:

    for word in generate_ngrams(tweet, n_gram=2):

        disaster_bigrams[word] += 1

        

for tweet in negative:

    for word in generate_ngrams(tweet, n_gram=2):

        nondisaster_bigrams[word] += 1

        

df_disaster_bigrams = pd.DataFrame(sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1])

df_nondisaster_bigrams = pd.DataFrame(sorted(nondisaster_bigrams.items(), key=lambda x: x[1])[::-1])



# Trigrams

disaster_trigrams = defaultdict(int)

nondisaster_trigrams = defaultdict(int)



for tweet in positive:

    for word in generate_ngrams(tweet, n_gram=3):

        disaster_trigrams[word] += 1

        

for tweet in negative:

    for word in generate_ngrams(tweet, n_gram=3):

        nondisaster_trigrams[word] += 1

        

df_disaster_trigrams = pd.DataFrame(sorted(disaster_trigrams.items(), key=lambda x: x[1])[::-1])

df_nondisaster_trigrams = pd.DataFrame(sorted(nondisaster_trigrams.items(), key=lambda x: x[1])[::-1])
fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)

plt.tight_layout()



sns.barplot(y=df_disaster_bigrams[0].values[:100], x=df_disaster_bigrams[1].values[:100], ax=axes[0], color='red')

sns.barplot(y=df_nondisaster_bigrams[0].values[:100], x=df_nondisaster_bigrams[1].values[:100], ax=axes[1], color='green')



for i in range(2):

    axes[i].spines['right'].set_visible(False)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    axes[i].tick_params(axis='x', labelsize=13)

    axes[i].tick_params(axis='y', labelsize=13)



N = 100

axes[0].set_title(f'Top {N} most common bigrams in Disaster Tweets', fontsize=15)

axes[1].set_title(f'Top {N} most common bigrams in Non-disaster Tweets', fontsize=15)



plt.show()
## adding bigram stopwords



add_bigrams_positive = ['in the','of the','on the','by the','at the','and the','to be','to the','by a','for the',

              'like a','of a','is a','from the','for a','as from','is from','for a','under a','in under','to a']



add_bigrams_negative = ['for a','i wa','do you','of a','have to',

                   'with the','be a','to get','it wa','are you','if i','full re','wa a']



stopwords_bigram = add_bigrams_positive + add_bigrams_negative

print(stopwords_bigram)
fig, axes = plt.subplots(ncols=2, figsize=(22, 50), dpi=100)

plt.tight_layout()



sns.barplot(y=df_disaster_trigrams[0].values[:100], x=df_disaster_trigrams[1].values[:100], ax=axes[0], color='red')

sns.barplot(y=df_nondisaster_trigrams[0].values[:100], x=df_nondisaster_trigrams[1].values[:100], ax=axes[1], color='green')



for i in range(2):

    axes[i].spines['right'].set_visible(False)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    axes[i].tick_params(axis='x', labelsize=13)

    axes[i].tick_params(axis='y', labelsize=13)



N = 100

axes[0].set_title(f'Top {N} most common trigrams in Disaster Tweets', fontsize=15)

axes[1].set_title(f'Top {N} most common trigrams in Non-disaster Tweets', fontsize=15)



plt.show()
### adding trigram stopwords



add_trigrams_positive = ['in under a','up by a']

add_trigrams_negative = ['china stock market','cross body bag','reddit will now']



stopwords_trigram = add_trigrams_positive + add_trigrams_negative

print(stopwords_trigram)
xtrain,xvalid,ytrain,yvalid = train_test_split(train.text.values, train.target.values, test_size=0.2, 

                                              random_state=22)

print(xtrain.shape)

print(xvalid.shape)
pd.DataFrame(xtrain).head()
stopwords = stopwords_unigram + stopwords_bigram + stopwords_trigram

ctv = CountVectorizer(stop_words=stopwords ,ngram_range=(1,3))



# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)

ctv.fit(list(xtrain) + list(xvalid))

xtrain_ctv =  ctv.transform(xtrain) 

xvalid_ctv = ctv.transform(xvalid)
# Fitting a simple Logistic Regression on Counts



clf = LogisticRegression(C=1.0)

clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict(xvalid_ctv)



print('Accuracy of Logistic Regression classifier on training set: {:.2f}'

     .format(clf.score(xtrain_ctv, ytrain)))

print('Accuracy of Logistic Regression classifier on test set: {:.2f}'

     .format(clf.score(xvalid_ctv, yvalid)))

print('F1-score of Logistic Regression is: {:.2f}'.format(f1_score(yvalid,predictions)))

cm = confusion_matrix(yvalid, predictions)

cm
tfv = TfidfVectorizer(min_df=2,  max_features=None, 

            strip_accents='unicode', analyzer='word',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



# Fitting TF-IDF to both training and test sets (semi-supervised learning)

tfv.fit(list(xtrain) + list(xvalid))

xtrain_tfv =  tfv.transform(xtrain) 

xvalid_tfv = tfv.transform(xvalid)
# Fitting a simple Logistic Regression on TFIDF

clf = LogisticRegression(C=1.0)

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict(xvalid_tfv)



print('Accuracy of Logistic Regression classifier(with tfidf) on training set: {:.2f}'

     .format(clf.score(xtrain_tfv, ytrain)))

print('Accuracy of Logistic Regression(with tfidf) classifier on test set: {:.2f}'

     .format(clf.score(xvalid_tfv, yvalid)))

print('F1-score of Logistic Regression is: {:.2f}'.format(f1_score(yvalid,predictions)))

cm = confusion_matrix(yvalid, predictions)

cm
# Fitting a simple Naive Bayes on TFIDF

clf = MultinomialNB()

clf.fit(xtrain_tfv, ytrain)

predictions = clf.predict(xvalid_tfv)



print('Accuracy of Naive Bayes classifier(with tfidf) on training set: {:.2f}'

     .format(clf.score(xtrain_tfv, ytrain)))

print('Accuracy of Naive Bayes classifier(with tfidf) classifier on validation set: {:.2f}'

     .format(clf.score(xvalid_tfv, yvalid)))

print('F1-score of Naive Bayes (with tf-idf) is: {:.2f}'.format(f1_score(yvalid,predictions)))

cm = confusion_matrix(yvalid, predictions)

cm
# Fitting a Naive Bayes on Counts



clf = MultinomialNB()

clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict(xvalid_ctv)



print('Accuracy of Naive Bayes classifier(with CountVectorizer) on training set: {:.2f}'

     .format(clf.score(xtrain_ctv, ytrain)))

print('Accuracy of Naive Bayes classifier(with CountVectorizer) on test set: {:.2f}'

     .format(clf.score(xvalid_ctv, yvalid)))

print('F1-score of Naive Bayes is: {:.2f}'.format(f1_score(yvalid,predictions)))

cm = confusion_matrix(yvalid, predictions)

cm
### There are more than 1lakh features obtained after CountVectorising

### We need to reduce it to make SVM run efficiently

svd = TruncatedSVD(n_components=120)

svd.fit(xtrain_tfv)

xtrain_svd = svd.transform(xtrain_tfv)

xvalid_svd = svd.transform(xvalid_tfv)



# Scale the data obtained from SVD. Renaming variable to reuse without scaling.

scl = StandardScaler()

scl.fit(xtrain_svd)

xtrain_svd_scl = scl.transform(xtrain_svd)

xvalid_svd_scl = scl.transform(xvalid_svd)
# Fitting a simple SVM( with tf-idf values)

clf = SVC(C=1.0) 

clf.fit(xtrain_svd_scl, ytrain)

predictions = clf.predict(xvalid_svd_scl)



print('Accuracy of SVC classifier(with TfidfVectorizer) on training set: {:.2f}'

     .format(clf.score(xtrain_svd_scl, ytrain)))

print('Accuracy of SVC classifier(with TfidfVectorizer) on test set: {:.2f}'

     .format(clf.score(xvalid_svd_scl, yvalid)))

print('F1-score of SVC classifier is: {:.2f}'.format(f1_score(yvalid,predictions)))

cm = confusion_matrix(yvalid, predictions)

cm
#### Fitting a simple SVM( with CountVectorizer fit)



svd.fit(xtrain_ctv)

xtrain_svd = svd.transform(xtrain_ctv)

xvalid_svd = svd.transform(xvalid_ctv)



# Scale the data obtained from SVD. Renaming variable to reuse without scaling.

scl = StandardScaler()

scl.fit(xtrain_svd)

xtrain_svd_scl = scl.transform(xtrain_svd)

xvalid_svd_scl = scl.transform(xvalid_svd)



clf = SVC(C=1.0) 

clf.fit(xtrain_svd_scl, ytrain)

predictions = clf.predict(xvalid_svd_scl)



print('Accuracy of SVC classifier(with CountVectorizer) on training set: {:.2f}'

     .format(clf.score(xtrain_svd_scl, ytrain)))

print('Accuracy of SVC classifier(with CountVectorizer) on test set: {:.2f}'

     .format(clf.score(xvalid_svd_scl, yvalid)))

print('F1-score of SVC classifier(over counts) is: {:.2f}'.format(f1_score(yvalid,predictions)))

cm = confusion_matrix(yvalid, predictions)

cm
# Fitting a simple xgboost on tf-idf

clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf.fit(xtrain_tfv.tocsc(), ytrain)

predictions = clf.predict(xvalid_tfv.tocsc())



print('Accuracy of XGB classifier(with TfidfVectorizer) on training set: {:.2f}'

     .format(clf.score(xtrain_tfv.tocsc(), ytrain)))

print('Accuracy of XGB classifier(with TfidfVectorizer) on test set: {:.2f}'

     .format(clf.score(xvalid_tfv.tocsc(), yvalid)))

print('F1-score of XGB classifier is: {:.2f}'.format(f1_score(yvalid,predictions)))

cm = confusion_matrix(yvalid, predictions)

cm
# Fitting a simple xgboost on Counts

clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf.fit(xtrain_ctv.tocsc(), ytrain)

predictions = clf.predict(xvalid_ctv.tocsc())



print('Accuracy of XGB classifier(with TfidfVectorizer) on training set: {:.2f}'

     .format(clf.score(xtrain_ctv.tocsc(), ytrain)))

print('Accuracy of XGB classifier(with TfidfVectorizer) on test set: {:.2f}'

     .format(clf.score(xvalid_ctv.tocsc(), yvalid)))

print('F1-score of XGB classifier is: {:.2f}'.format(f1_score(yvalid,predictions)))

cm = confusion_matrix(yvalid, predictions)

cm
### parameter(C) tuning using Grid Search for Logistic regression over counts



gsc = GridSearchCV(estimator=LogisticRegression(max_iter=1000), 

                  param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000]},

                  cv=5, scoring='f1')



grid_result = gsc.fit(xtrain_ctv, ytrain)

best_params = grid_result.best_params_

best_params
### parameter(C) tuning using Grid Search for NaiveBayes over counts



gsc = GridSearchCV(estimator=MultinomialNB(), 

                  param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},

                  cv=5, scoring='f1')



grid_result = gsc.fit(xtrain_ctv, ytrain)

best_params = grid_result.best_params_

best_params
# Initialize SVD

svd = TruncatedSVD()

    

# Initialize the standard scaler 

scl = StandardScaler()



# We will use logistic regression here..

lr_model = LogisticRegression(max_iter=1000)



# Create the pipeline 

clf = Pipeline([('svd', svd),

                         ('scl', scl),('lr',lr_model)])
param_grid = {'svd__n_components' : [120, 180],

              'lr__C': [0.1, 1.0, 10], 

              'lr__penalty': ['l1', 'l2']}
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1',

                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)



result = model.fit(xtrain_ctv, ytrain)

best_params = result.best_params_
# hence the best fit happens with 180 svd components

best_params
svd = TruncatedSVD(n_components=180)

svd.fit(xtrain_ctv)

xtrain_svd = svd.transform(xtrain_ctv)

xvalid_svd = svd.transform(xvalid_ctv)



clf = LogisticRegression(C=1.0, max_iter=1000, penalty='l2')

clf.fit(xtrain_svd, ytrain)

predictions = clf.predict(xvalid_svd)



print('Accuracy of Logistic Regression classifier(with counts) on training set: {:.2f}'

     .format(clf.score(xtrain_svd, ytrain)))

print('Accuracy of Logistic Regression(with counts) classifier on test set: {:.2f}'

     .format(clf.score(xvalid_svd, yvalid)))

print('F1-score of Logistic Regression is: {:.2f}'.format(f1_score(yvalid,predictions)))

cm = confusion_matrix(yvalid, predictions)

cm
# Model tuning a RandomForestClassifier with 180 SVD components



gsc = GridSearchCV(estimator=RandomForestClassifier(), 

                  param_grid = {'n_estimators': [10, 100, 200]},

                  cv=5, scoring='f1')



grid_result = gsc.fit(xtrain_svd, ytrain)

best_params = grid_result.best_params_

best_params
# Fitting the RandomForest classifier with 180 svd components (over counts)



clf = RandomForestClassifier(n_estimators=100)

clf.fit(xtrain_svd, ytrain)

predictions = clf.predict(xvalid_svd)

print('F1-score of RandomForest is: {:.2f}'.format(f1_score(yvalid,predictions)))
# Model tuning a RidgeClassifier with 180 SVD components



gsc = GridSearchCV(estimator=RidgeClassifier(), 

                  param_grid = {'alpha': [0.01, 0.1, 1, 10, 50, 100]},

                  cv=5, scoring='f1')



grid_result = gsc.fit(xtrain_ctv, ytrain)

best_params = grid_result.best_params_

best_params
clf = RidgeClassifier(alpha=10)

clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict(xvalid_ctv)



print('Accuracy of Ridge Regression classifier(with counts) on training set: {:.2f}'

     .format(clf.score(xtrain_ctv, ytrain)))

print('Accuracy of Ridge Regression classifier(with counts) classifier on test set: {:.2f}'

     .format(clf.score(xvalid_ctv, yvalid)))

print('F1-score of Ridge Regression is: {:.2f}'.format(f1_score(yvalid,predictions)))
# Input shape for model

input_shape = xtrain_tfv.shape[1]



# Set callbacks to avoid overfitting

callbacks = [EarlyStopping(patience=3), ReduceLROnPlateau(patience=2)]
# Construct multilayer net



mlp_model = Sequential([

    Dense(64, input_dim=input_shape, activation='relu'),

    Dropout(0.5),

    Dense(32, activation='relu'),

    Dropout(0.5),

    Dense(16, activation='relu'),

    Dropout(0.5),

    Dense(1, activation='sigmoid'),   

])



mlp_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
pd.DataFrame(xvalid_ctv.toarray())
# Train model

#train_sparse = pd.concat(pd.DataFrame(xtrain_ftv))

#tf.sparse.reorder(xtrain.tfv);tf.

mlp_model.fit(xtrain_tfv.toarray(), ytrain,

             batch_size=16,

             epochs=30, 

             verbose=2,

             callbacks=callbacks,

             validation_data=(xvalid_tfv.toarray(), yvalid))
# Check model on test validation_set

test_score = mlp_model.evaluate(xvalid_tfv.toarray(), yvalid)

print(test_score)
### cleaning the test texts

test["text"] = test["text"].apply(lambda x:remove_irrelevant(x))

test["text"] = test["text"].apply(lambda x:remove_punc(x))

test["text"] = test["text"].apply(lambda x:remove_emoji(x))

test["text"] = test["text"].apply(lambda x:unique_char(rep,x))

test["text"] = test["text"].apply(lambda x:lemmatize(x))

my_final_submission = pd.DataFrame({'id':test.id.values, 'text':test.text.values})
xtest_tfv =  tfv.transform(my_final_submission.text.values)

my_final_submission["target"] = mlp_model.predict_classes(xtest_tfv.toarray())

my_final_submission.drop(columns=['text'], inplace=True)

my_final_submission.head()
#clf = LogisticRegression(max_iter=1000, C=1.0)

#clf.fit(xtrain_ctv, ytrain)
#xtest_ctv =  ctv.transform(my_final_submission.text.values)

#my_final_submission["target"] = clf.predict(xtest_ctv)

#my_final_submission.drop(columns=['text'], inplace=True)

#my_final_submission.head()
my_final_submission.to_csv("submission.csv", index=False)