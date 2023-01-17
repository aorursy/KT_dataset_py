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
data=pd.read_csv('../input/amazon-music-reviews/Musical_instruments_reviews.csv')
data.head()
print('Total ratings per rating:','\n',data.overall.value_counts())

#Number of unique instrument ids
print('Number of unique instruments:',len(data.asin.unique()))
print('Number of rows:',data.shape[0])      
#combine text and summary columns
data['reviews'] = data['reviewText'] + ' ' + data['summary']
del data['reviewText'] 
del data['summary']

#rename overall to rating
data.rename(columns={'overall':'rating'},inplace=True)
data['reviews'].isnull().sum()
#drop rows with missing reviews
data.dropna(axis=0, inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#look at top 20 instrument reviews
#and bottom 20 instrument reviews

top_20=data.asin.value_counts().head(20)
btm_20 = data.asin.value_counts().tail(20)
#Create pivot table to plot
top_20_df=pd.DataFrame()
top_20_ids=list(top_20.index)
for i in top_20_ids:
    top_20_df=top_20_df.append(data[data['asin']==i],ignore_index=True)
table = pd.pivot_table(top_20_df, values='rating',index=top_20_df['asin'],aggfunc=np.mean)

#Create Figure
plt.figure(figsize=(10,6))
sns.barplot(x=table.index, y='rating', data=table)
plt.xticks(rotation=90)
plt.xlabel('Instrument ID')
plt.ylabel('Average Rating')
plt.title('Instruments with the Highest Number of Ratings (Top 20)')
plt.tight_layout()
plt.show()
#Plot ave rating for 20 Instruments with lower number of ratings
btm_20_df=pd.DataFrame()
btm_20_ids=list(btm_20.index)
for i in btm_20_ids:
    btm_20_df=btm_20_df.append(data[data['asin']==i],ignore_index=True)
table_btm = pd.pivot_table(btm_20_df, values='rating',index=btm_20_df['asin'],aggfunc=np.mean)

plt.figure(figsize=(10,6))
sns.barplot(x=table_btm.index, y='rating', data=table_btm)
plt.xticks(rotation=90)
plt.xlabel('Instrument ID')
plt.ylabel('Average Rating for Instrument')
plt.title('Instruments with Fewest Number of Ratings (Bottom 20)')
plt.tight_layout()
plt.show()
#Plot ratings percentages
t=pd.DataFrame(data=data['rating'].value_counts(normalize=True)*100)
plt.figure(figsize=(10,6))
sns.barplot(x=t.index, y=t.rating,palette="Blues_d")
plt.xlabel('Rating',fontsize=20)
plt.ylabel('Percent of Total Ratings',fontsize=20)
plt.show()
#drop columns not using for analysis
col_to_drop=['reviewerID','asin','reviewerName','helpful','unixReviewTime','reviewTime']
instrument_reviews=data.drop(columns=col_to_drop, axis=1)
instrument_reviews.head()
#Create sentiment column 
instrument_reviews['sentiment'] = instrument_reviews['rating'].map({5:2,4:2,3:1,2:0,1:0})   
instrument_reviews.head()
instrument_reviews.sentiment.value_counts(normalize=True)*100
# Data to plot
labels=['Positive','Neutral','Negative']
sizes = [instrument_reviews['sentiment'].value_counts(normalize=True)]
labels_rating = ['5','4','3','2','1']
sizes_rating = [instrument_reviews['rating'].value_counts(normalize=True)]
#colors = ['olive','lightcoral']
#colors_rating = ['blue','cyan', 'purple','gray']
colors_rating=sns.color_palette("BuGn_r")
colors=sns.color_palette("PuRd")
explode = (0.1,0.1,0.1) 
explode_ratings = (0.1,0.1,0.1,0.1,0.1)

# Plot
plt.pie(sizes, labels=labels, colors=colors, startangle=90,frame=True,explode=explode)
plt.pie(sizes_rating,labels=labels_rating,colors=colors_rating,radius=0.75,startangle=90,explode=explode_ratings)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
 
plt.axis('equal')
plt.tight_layout()
plt.show()
plt.figure(figsize=(10,6))
colors=['olive','yellow','lightcoral']
 
plt.pie(instrument_reviews['sentiment'].value_counts(normalize=True),colors=colors,labels=['Positive','Neutral','Negative'],autopct='%1.2f%%',shadow=True)
plt.title('Sentiment',fontsize=20)

plt.show()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

#import models to test
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

#import metrics
from sklearn.metrics import accuracy_score, classification_report

import string
import re
import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()
#Splitting into train and valid
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(instrument_reviews['reviews'], instrument_reviews['sentiment'], random_state=42, test_size=0.3)
print('Training Data Shape:', X_train.shape)
print('Testing Data Shape:', X_valid.shape)
#Define functions to clean and tokenize data
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”",".",""]

class CleanTextTransformer(TransformerMixin):
   def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]
   def fit(self, X, y=None, **fit_params):
        return self
def get_params(self, deep=True):
        return {}
    
def cleanText(text):    
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text

def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    return tokens
#Create string of Positive, Neutral, Negative words for Wordcloud
pos = X_train[y_train[y_train == 2].index]
neut = X_train[y_train[y_train == 1].index]
neg = X_train[y_train[y_train == 0].index]

X_train.shape,pos.shape,neut.shape,neg.shape
#Create text for wordcloud for each sentiment
pos_words=''
for w in pos.apply(cleanText).apply(tokenizeText):
     pos_words+=" ".join(w)
print('There are {} positive words'.format(len(pos_words)))

neut_words=''
for w in neut.apply(cleanText).apply(tokenizeText):
     neut_words+=" ".join(w)
print('There are {} neutral words'.format(len(neut_words)))        

neg_words=''
for w in neg.apply(cleanText).apply(tokenizeText):
     neg_words+=" ".join(w)
print('There are {} negative words'.format(len(neg_words)))        

#Negative wordcloud
from wordcloud import WordCloud
plt.figure(figsize = (16,16)) 
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(neg_words)
plt.imshow(wc,interpolation = 'bilinear')
plt.show()
#Neutral wordcloud
plt.figure(figsize = (16,16)) 
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(neut_words)
plt.imshow(wc,interpolation = 'bilinear')
plt.show()
#Positive wordcloud
plt.figure(figsize = (16,16)) 
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(pos_words)
plt.imshow(wc,interpolation = 'bilinear')
plt.show()
#Models to test
clf = LinearSVC(max_iter=10000)
xgb = XGBClassifier(n_estimators = 100, learning_rate=0.1)
rfc = RandomForestClassifier(n_estimators=100)
lr = LogisticRegression(max_iter=500)
mnb = MultinomialNB()

models = [clf, xgb, rfc, lr, mnb]
# def printNMostInformative(vectorizer, clf, N):
#     feature_names = vectorizer.get_feature_names()
#     coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
#     topNeg = coefs_with_fns[:N]
#     topPos = coefs_with_fns[:-(N + 1):-1]
#     print("Negative best: ")
#     for feat in topNeg:
#         print(feat)
#     print("Positive best: ")
#     for feat in topPos:
#         print(feat)
# print("Top 10 features used to predict: ")        
# printNMostInformative(vectorizer, clf, 10)
# pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
# transform = pipe.fit_transform(X_train, y_train)
# vocab = vectorizer.get_feature_names()
# for i in range(len(X_train)):
#     s = ""
#     indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
#     numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
#     for idx, num in zip(indexIntoVocab, numOccurences):
#         s += str((vocab[idx], num))
#Create loop to get accuracy and classification report for models
for model in models:
    
    vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)
    print('model:',model,'\t',"accuracy:", accuracy_score(y_valid, preds))
    print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')

model = XGBClassifier(subsample=0.8)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__n_estimators': range(100,1001,300),
    'model__max_depth': [8],
    'model__learning_rate':[0.1]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)
model = XGBClassifier(subsample=0.8)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__n_estimators': range(100,1001,300),
    'model__max_depth': [5],
    'model__learning_rate':[0.1]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)
model = XGBClassifier(subsample=0.8)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__n_estimators': [700],
    'model__max_depth': [5,8],
    'model__learning_rate':[0.1,0.2]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)
model = MultinomialNB()

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__alpha': np.linspace(0.5,1.6,6),
    'model__fit_prior': [True, False]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)
model = RandomForestClassifier()

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__n_estimators': [100],
    'model__max_features': ['auto'],
    'model__max_depth' : [2,3,4],
    'model__criterion' :['gini'],
    'model__class_weight': [None]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)
model = LogisticRegression(max_iter=500)
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
# pipe.fit(X_train, y_train)
# preds = pipe.predict(X_valid)
# print("accuracy:", accuracy_score(y_valid, preds))
# print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')

#
param_grid = {'model__C': (0.01,0.1,1),'model__class_weight': [None,'balanced']}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)
####optimized models
lr_opt = LogisticRegression(max_iter=500,C=1,class_weight=None)
rf_opt = RandomForestClassifier(n_estimators=100, max_features='auto',max_depth=4,class_weight=None)
xgb_opt = XGBClassifier(subsample=0.8, learning_rate = 0.1, n_estimators = 700, max_depth = 5 )
mnb_opt = MultinomialNB(alpha = 0.72, fit_prior = False)
models_opt = [lr_opt, rf_opt, xgb_opt, mnb_opt]
#LogisticRegressor
model = lr_opt
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_valid)
lr_opt_acc = accuracy_score(y_valid, preds)
print('model:',model,'\t',"accuracy:",lr_opt_acc )
print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')

#RandomForestClassifier
model = rf_opt
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_valid)
rf_opt_acc = accuracy_score(y_valid, preds)
print('model:',model,'\t',"accuracy:",rf_opt_acc )
print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')
#XGBClassifier
model = xgb_opt
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_valid)
xgb_opt_acc = accuracy_score(y_valid, preds)
print('model:',model,'\t',"accuracy:",xgb_opt_acc )
print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')
#MultinomialNB
model = mnb_opt
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_valid)
mnb_opt_acc = accuracy_score(y_valid, preds)
print('model:',model,'\t',"accuracy:",mnb_opt_acc )
print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')
model_name = ['Logistic Regression', 'Random Forest Classifier', 'XGB Classifier', 'Multinomial NB']
accuracy = [lr_opt_acc*100, rf_opt_acc*100,xgb_opt_acc*100,mnb_opt_acc*100]

plt.figure(figsize=(10,5))

plt.ylabel("Test Accuracy %",fontsize=14)
plt.xlabel("Machine Learning Model",fontsize=14)
sns.lineplot(x= model_name, y= accuracy)
plt.show()
##Find top 15 most frequent words in the text for each sentiment
import itertools
import collections
d=pos.apply(cleanText).apply(tokenizeText)

# List of all positive words 
pos_list = list(itertools.chain(*d))

# Create counter
counts_pos = collections.Counter(pos_list)
print('Most common positive words:')
items, counts = zip(*counts_pos.most_common(15))
pd.Series(counts, index=items)
pd.Series(counts, index=items).to_frame()
d=neut.apply(cleanText).apply(tokenizeText)
# List of all neutral words
neut_list = list(itertools.chain(*d))

# Create counter
counts_neut = collections.Counter(neut_list)

print('Most common neutral words:')
items, counts = zip(*counts_neut.most_common(15))
pd.Series(counts, index=items)
pd.Series(counts, index=items).to_frame()      
d=neg.apply(cleanText).apply(tokenizeText)
# List of all negative words
neg_list = list(itertools.chain(*d))

# Create counter
counts_neg = collections.Counter(neg_list)

print('Most common negative words:')
items, counts = zip(*counts_neg.most_common(15))
pd.Series(counts, index=items)
pd.Series(counts, index=items).to_frame() 
#Define functions to clean and tokenize data adding list of words to remove
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”",".",""]
REM = ["guitar","string","strings","amp","pedal","\'s"]
class CleanTextTransformer(TransformerMixin):
   def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]
   def fit(self, X, y=None, **fit_params):
        return self
def get_params(self, deep=True):
        return {}
    
def cleanText(text):    
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text

def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    tokens = [tok for tok in tokens if tok not in REM]
    return tokens
#Splitting into train and valid
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(instrument_reviews['reviews'], instrument_reviews['sentiment'], random_state=42, test_size=0.3)
print('Training Data Shape:', X_train.shape)
print('Testing Data Shape:', X_valid.shape)
#LogisticRegressor with common words removed
model = lr_opt
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_valid)
lr_opt_acc_filtered = accuracy_score(y_valid, preds)
print('model:',model,'\t',"accuracy:",lr_opt_acc_filtered )
print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')
#RandomForestClassifier
model = rf_opt
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_valid)
rf_opt_acc_filtered = accuracy_score(y_valid, preds)
print('model:',model,'\t',"accuracy:",rf_opt_acc_filtered )
print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')
#XGBClassifier
model = xgb_opt
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_valid)
xgb_opt_acc_filtered = accuracy_score(y_valid, preds)
print('model:',model,'\t',"accuracy:",xgb_opt_acc_filtered )
print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')
#MultinomialNB
model = mnb_opt
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_valid)
mnb_opt_acc_filtered = accuracy_score(y_valid, preds)
print('model:',model,'\t',"accuracy:",mnb_opt_acc_filtered )
print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')
#plot of nonfiltered and filtered
model_name = ['Logistic Regression', 'Random Forest Classifier', 'XGB Classifier', 'Multinomial NB']
accuracy = [lr_opt_acc*100, rf_opt_acc*100,xgb_opt_acc*100,mnb_opt_acc*100]
accuracy_filtered = [lr_opt_acc_filtered*100, rf_opt_acc_filtered*100,xgb_opt_acc_filtered*100,mnb_opt_acc_filtered*100]
plt.figure(figsize=(10,5))

plt.ylabel("Test Accuracy %",fontsize=14)
plt.xlabel("Machine Learning Model",fontsize=14)
sns.lineplot(x= model_name, y= accuracy,label='Not Filtered')
sns.lineplot(x= model_name, y= accuracy_filtered,label='Filtered')
plt.show()
