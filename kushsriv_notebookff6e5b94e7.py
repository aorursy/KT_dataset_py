import nltk
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
import seaborn as sns
import string
train = pd.read_csv('../input/twitter-disaster-dataset/train.csv')
train.head()
train.shape
test = pd.read_csv('../input/twitter-disaster-dataset/test.csv')
test.head()
test['keyword'].isnull().sum()
#create different set for True and False train sets
train_true = train[train['target']==1]
train_true.shape
train_false = train[train['target']==0]
train_false.shape
#keywords used in true cases
keyword_true = train_true['keyword'].value_counts()
keyword_true
#keywords used in false cases
keyword_false = train_false['keyword'].value_counts()
keyword_false
df_all = pd.concat([train,test],axis=0)
df_all.head()
sns.heatmap(df_all.isnull(),yticklabels=False,cbar=False)
df_all.isnull().sum()
#new feature length of text
df_all['text_len'] = df_all['text'].apply(lambda x : len(x))
df_all['text_len'].head()
def capital_letters_count(sentence):
    sum1 = 0
    for c in sentence:
        if c.isupper():
            sum1 = sum1 + 1
    return sum1
# number of capital letters
df_all['capital_letters'] = df_all['text'].apply(lambda x : capital_letters_count(x))
df_all['capital_letters'].head()
df_all['capital_letters'].value_counts()
string.punctuation

def count_punctuation(sentence):
    sum2 = 0 
    for c in sentence:
        if c in string.punctuation:
            sum2 = sum2 + 1
    return sum2
            
        
#feature - number of punctuations
df_all['punctuations'] = df_all.text.apply(lambda x : count_punctuation(x))
df_all['punctuations'].head()
df_all['location'].dtypes
df_all['location_text'] = df_all['location'].astype(str) + ' ' + df_all['text'].astype(str)
df_all['location_text'].head()
df_all.head()
df_all['keyword_text'] = df_all['keyword'].astype(str) + ' ' + df_all['text'].astype(str)
df_all.head()
df_all.head()
#removing punctuation and stop words function
def clean_text(sentence):
    ps = nltk.PorterStemmer()
    #nltk.download("stopwords")
    chachedWords = stopwords.words('english')
    chachedWords.append('nan')
    temp = ""
    return_sentence = ""
    sentence = sentence.lower()
    for c in sentence:
        if c not in string.punctuation:
            temp = temp+c
    temp_arr = temp.split()
    for word in temp_arr:
        if word not in chachedWords:
            word = ps.stem(word)
            return_sentence = return_sentence + word + " "
    return_sentence = return_sentence.strip()
    return return_sentence
nltk.download("stopwords")
nan = ['nan']
df_all['clean_text'] = df_all.keyword_text.apply(lambda x: clean_text(x))
df_all['clean_text'].head()
#only remove puntuation
def remove_punctuation(sentence):
    ps = nltk.PorterStemmer()
    null_value = ['nan']
    temp = ""
    return_sentence = ""
    sentence = sentence.lower()
    for c in sentence:
        if c not in string.punctuation:
            temp = temp+c
    temp_arr = temp.split()
    for word in temp_arr:
        if word not in null_value:
            return_sentence = return_sentence + word + " "
    return_sentence = return_sentence.strip()
    return return_sentence
#only removed punctuations from the text this will be used to fill missing values in 'keyword' columns
df_all['remove_punctuations'] = df_all.text.apply(lambda x : remove_punctuation(x))
df_all['remove_punctuations'].head()
df_all.describe()

pd.set_option('display.max_rows', 221)
df['keyword'].value_counts()
keywords = df['keyword'].unique()
keywords = np.delete(keywords,0) # delete nan keyword
keywords
all_keywords = []
def fill_missing_keyword_values(sentence):
    temp = sentence.split()
    return_word = np.nan
    for word in temp:
        if word in keywords:
            return_word = word
            break
    print(return_word)
    return return_word
            
    
df['keyword'].isnull().sum()
#df['keyword'] = df.apply(lambda row: fill_missing_keyword_values(row['remove_punctuations']) if np.isnan(row['keyword']) else row['keyword'])
#df_all_feature.isnull().sum()
#creating a new DF
df = df_all.drop(['location','text','location_text','keyword_text','remove_punctuations'],axis=1)
df.head()
df_scaled = df.copy()
#scale data using standard scaler
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
df_scaled [['text_len','capital_letters','punctuations']]= scale.fit_transform(df_scaled[['text_len','capital_letters','punctuations']])
df_scaled.head()
#Encoding data using Count Venctorizer
count_vect = CountVectorizer()
X_count = count_vect.fit_transform(df['clean_text'])
df.reset_index(drop=True, inplace=True)
df_scaled.reset_index(drop=True, inplace=True)
X_count_feat_scaled = pd.concat([df_scaled['target'],df_scaled['text_len'],df_scaled['capital_letters'],df_scaled['punctuations'],
                           pd.DataFrame(X_count.toarray())], axis=1)
X_count_feat = pd.concat([df['target'],df['text_len'],df['capital_letters'],df['punctuations'],
                           pd.DataFrame(X_count.toarray())], axis=1)
X_count_feat.head()
X_count_feat_scaled.head()
#TF-IDF
tfidf_vect = TfidfVectorizer()
X_tfidf = tfidf_vect.fit_transform(df['clean_text'])
X_tfidf_feat_scaled = pd.concat([df_scaled['target'],df_scaled['text_len'],df_scaled['capital_letters'],
                          df_scaled['punctuations'],pd.DataFrame(X_tfidf.toarray())], axis=1)
X_tfidf_feat = pd.concat([df['target'],df['text_len'],df['capital_letters'],
                          df['punctuations'],pd.DataFrame(X_tfidf.toarray())], axis=1)
X_tfidf_feat.head()
del df
del df_scaled
del train
del test
del df_all
del train_true
del train_false
del keyword_true
del keyword_false
import gc
gc.collect()
X_tfidf_feat_scaled.head()
train_count = X_count_feat[X_count_feat.target.notnull()]
train_count.shape
test_count = X_count_feat[X_count_feat.target.isnull()]
test_count.shape
train_count_scaled = X_count_feat_scaled[X_count_feat_scaled.target.notnull()]
train_count_scaled.shape
test_count_scaled = X_count_feat_scaled[X_count_feat_scaled.target.isnull()]
test_count_scaled.shape
del X_count_feat
del X_count_feat_scaled
#training sets for count vectorizer
x_train_count = train_count.drop(['target'],axis=1)
y_train_count = train_count['target']
#testing set for count vectorizer
x_test_count = test_count.drop(['target'],axis=1)
del train_count
del test_count
gc.collect()
#training sets for count vectorizer for scaled data 
x_train_count_scaled = train_count_scaled.drop(['target'],axis=1)
y_train_count_scaled = train_count_scaled['target']
#testing set for count vectorizer for scaled data
x_test_count_scaled = test_count_scaled.drop(['target'],axis=1)
del test_count_scaled
del train_count_scaled
gc.collect()
train_tfidf = X_tfidf_feat[X_tfidf_feat.target.notnull()]
train_tfidf.shape
test_tfidf = X_tfidf_feat[X_tfidf_feat.target.isnull()]
train_tfidf.shape
del X_tfidf_feat
gc.collect()
#train data setf for tfidf
x_train_tfidf = train_tfidf.drop(['target'],axis=1)
y_train_tfidf = train_tfidf['target']

#test for tfidf
x_test_tfidf =test_tfidf.drop(['target'],axis=1)
del train_tfidf
del test_tfidf
gc.collect()
train_tfidf_scaled = X_tfidf_feat_scaled[X_tfidf_feat_scaled.target.notnull()]
train_tfidf_scaled.shape
test_tfidf_scaled = X_tfidf_feat_scaled[X_tfidf_feat_scaled.target.isnull()]
train_tfidf_scaled.shape