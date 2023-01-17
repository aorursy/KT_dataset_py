# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/amazon-music-reviews/Musical_instruments_reviews.csv')
df_copy = df.copy()
df.head(3)
df.reviewText   = df.reviewText + df.summary
df.reviewTime   = df.reviewTime.apply(lambda string: [int(i) for i in string.replace(',','').split()])
df['month']     = df.reviewTime.apply(lambda x: x[0])
df['date']      = df.reviewTime.apply(lambda x: x[1])
df['year']      = df.reviewTime.apply(lambda x: x[2])
df.drop(columns = ['asin','helpful','summary','unixReviewTime','reviewTime'],axis=0,inplace=True)

df.head(3)
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import wordnet,WordNetLemmatizer

def remove_punctuation(the_string):
    for c in string.punctuation:
        the_string = str(the_string).replace(c,'')
    return the_string

def remove_digits(the_string):
    for c in range(10):        
        the_string = str(the_string).replace(str(c),'')
    return the_string

df.reviewText = df.reviewText.apply(remove_punctuation).apply(remove_digits)
def remove_stopwords(sentence):
    stopword_list = stopwords.words('english')
    stopword_list.append(['www','http'])
    new_sentence = ''
    for word in sentence.split():
        if word not in stopword_list:
            new_sentence += ' '+word.lower()
    return new_sentence[1:]

df.reviewText = df.reviewText.apply(remove_stopwords)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

wnl = WordNetLemmatizer()

def do_lemmatize(sentence):
      
    _list = nltk.pos_tag(str(sentence).split())   
    
    the_sentence = ''
    for _tuple in _list:        
        wrd    = _tuple[0]                      
        if _tuple[1][0] in ['N','V','J','R']:
            if _tuple[1][0]=='N':
                pos_tg = 'n'
            elif _tuple[1][0]=='V':
                pos_tg = 'v'
            elif _tuple[1][0]=='J':
                pos_tg = 'a'
            else:
                pos_tg = 'r'
        else:
            pos_tg = 'n'
            
        the_sentence+=' ' + wnl.lemmatize(wrd,pos_tg)
        
    return the_sentence[1:]

df.reviewText = df.reviewText.apply(do_lemmatize)

del wnl

df.head(3)
from nltk.stem import SnowballStemmer

sbs = SnowballStemmer('english')

def stem_tokens(sentence):
    the_sentence = ''
    for word in str(sentence).split():
        the_sentence+=' '+sbs.stem(word)
    return the_sentence

df.reviewText = df.reviewText.apply(stem_tokens)

del sbs

df.head(3)
df['n_words']      = df.reviewText.apply(lambda x:len(x))
df['unique_words'] = df.reviewText.apply(lambda string:len(set(str(string).split())))

df.head(3)
from textblob import TextBlob

def get_sentiment(string):
    return list(TextBlob(string).sentiment)

df['tb_sentiment'] = df.reviewText.apply(get_sentiment)
df['polarity']     = df.tb_sentiment.apply(lambda x:x[0])
df['subjectivity'] = df.tb_sentiment.apply(lambda x:x[1])
df.drop(columns=['tb_sentiment'],axis=0,inplace=True)
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

sentence_list = []
for index,row in df.iterrows():    
    sentence_list.append(row.reviewText)    
        
V = TfidfVectorizer()
_tuple   = V.fit_transform(sentence_list)
col_name = V.get_feature_names()  

del sentence_list
temp = pd.DataFrame(_tuple)
temp.columns=['dat']

def get_tfidf_disp(temp):
    array = list([])
    for i in range(temp.shape[0]):
        sentence_list = str(temp.dat[i]).split('\n')
        for sentence in sentence_list:
            word_list = sentence.split('\t')        
            
            word_list[0] = word_list[0].replace(',','').replace('(','').replace(')','').replace(':','')            
            
            _temp = word_list[0].split()
            
            if len(_temp)>0:
                word_id = word_list[0].split()[1]                                
                value = word_list[1]
                array.append([i,word_id,float(value)])
                                        
    array = pd.DataFrame(array)
    array.columns = ['doc','word_id','value']
    array.sort_values(by='value',ascending=False,inplace=True)
    return array

temp_array = get_tfidf_disp(temp)
del temp

temp_array.head(3)              
#temp_array = temp_array[:200]
select_col = list(set([col_name[int(word_id)] for word_id in temp_array.word_id]))

del temp_array

print(*select_col)
array   = _tuple.toarray()
temp_df = pd.DataFrame(array)
temp_df.columns = col_name

del array

temp_df = temp_df[select_col]
temp_df.head(3)
for col in temp_df.columns:
    df[col] = temp_df[col]
df.drop(columns=['reviewText'],axis=0,inplace=True)    

del temp_df

df.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def give_prediction(temp_df):
   
    y = [1 if i==5 else 0 for i in temp_df.overall]        
    X = temp_df.drop(columns='overall',axis=0)
    
    # Train-Test Split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=7)
    print('sum of y-test = ',sum(y_test),len(y_test))

    # Drop Unnecessary Columns before Training
    X_train_id = X_train[['reviewerID','reviewerName']]
    X_train.drop(columns=['reviewerID','reviewerName'],axis=0,inplace=True)

    X_test_id = X_test[['reviewerID','reviewerName']]
    X_test.drop(columns=['reviewerID','reviewerName'],axis=0,inplace=True)
    
    # Model Development
    lr = LogisticRegression(max_iter=100, solver='liblinear',random_state=7)
    lr.fit(X_train,y_train)
    
    # Get Predictions  
    y_pred = lr.predict(X_test) 
    y_pred_proba = lr.predict_proba(X_test)
    X_test_id['y_test']       = y_test
    X_test_id['y_pred']       = y_pred
    X_test_id['y_pred_proba'] = [r[0] for r in y_pred_proba]
  
    return X_test_id,lr.coef_


df_345 = df.copy()
result_df,coef = give_prediction(df_345)

print('Accuracy Score = '        ,accuracy_score(result_df.y_test,result_df.y_pred))
print('\nConfusion Matrix\n'     ,confusion_matrix(result_df.y_test,result_df.y_pred))
print('\nClassification Report\n',classification_report(result_df.y_test,result_df.y_pred))

result_df.y_pred.hist(bins=2)

del df_345,result_df,coef
df_345 = df[df.overall>2]
result_df,coef_ = give_prediction(df_345)

print('Accuracy Score = '        ,accuracy_score(result_df.y_test,result_df.y_pred))
print('\nConfusion Matrix\n'     ,confusion_matrix(result_df.y_test,result_df.y_pred))
print('\nClassification Report\n',classification_report(result_df.y_test,result_df.y_pred))

result_df.y_pred.hist(bins=2)

del df_345,result_df,coef_
df_45 = df[df.overall>3]
result_df,coef_ = give_prediction(df_45)

print('Accuracy Score = '        ,accuracy_score(result_df.y_test,result_df.y_pred))
print('\nConfusion Matrix\n'     ,confusion_matrix(result_df.y_test,result_df.y_pred))
print('\nClassification Report\n',classification_report(result_df.y_test,result_df.y_pred))

result_df.y_pred.hist(bins=2)

del df_45,result_df,coef_
df_word = df.copy()
df_word['nu_words'] = df_word.n_words - df_word.unique_words
df_word.drop(columns=['unique_words','month','date','year'],axis=0,inplace=True)

result_df,coef_ = give_prediction(df_word)

print('Accuracy Score = '        ,accuracy_score(result_df.y_test,result_df.y_pred))
print('\nConfusion Matrix\n'     ,confusion_matrix(result_df.y_test,result_df.y_pred))
print('\nClassification Report\n',classification_report(result_df.y_test,result_df.y_pred))

result_df.y_pred.hist(bins=2)

del df_word,result_df,coef_
df_us = df[df.overall==5][:5000]
df_us = df_us.append(df[df.overall<5])

result_df,coef_ = give_prediction(df_us)

print('Accuracy Score = '        ,accuracy_score(result_df.y_test,result_df.y_pred))
print('\nConfusion Matrix\n'     ,confusion_matrix(result_df.y_test,result_df.y_pred))
print('\nClassification Report\n',classification_report(result_df.y_test,result_df.y_pred))

result_df.y_pred.hist(bins=2)

del df_us,result_df,coef_
df_os = df.copy()
for i in range(1):
    df_os = df.append(df[df.overall<2])

result_df,coef_ = give_prediction(df_os)

print('Accuracy Score = '        ,accuracy_score(result_df.y_test,result_df.y_pred))
print('\nConfusion Matrix\n'     ,confusion_matrix(result_df.y_test,result_df.y_pred))
print('\nClassification Report\n',classification_report(result_df.y_test,result_df.y_pred))

result_df.y_pred.hist(bins=2)

del df_os,result_df,coef_
#df_cb = df.copy()

# Under Sampling
df_cb = df[df.overall==5][:]
df_cb = df_cb.append(df[df.overall<5])

# Feature Engineering
df_cb['nu_words'] = df_cb.n_words - df_cb.unique_words
df_cb.drop(columns=['unique_words','month','date','year'],axis=0,inplace=True)

# Over Sampling
for i in range(0):
    df_cb = df_cb.append(df_cb[df_cb.overall<2])
    
# Predictive Analysis    
result_df,coef_ = give_prediction(df_cb)

print('Accuracy Score = '        ,accuracy_score(result_df.y_test,result_df.y_pred))
print('\nConfusion Matrix\n'     ,confusion_matrix(result_df.y_test,result_df.y_pred))
print('\nClassification Report\n',classification_report(result_df.y_test,result_df.y_pred))

result_df.y_pred.hist(bins=2)

del df_cb,result_df,coef_

