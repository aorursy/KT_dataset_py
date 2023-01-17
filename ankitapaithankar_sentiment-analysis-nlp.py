import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

## read train and test data

train=pd.read_csv("../input/identifying-the-sentiments-nlp/train_2kmZucJ.csv")
train.head()
train.shape
test=pd.read_csv("../input/identifying-the-sentiments-nlp/test_oJQbWVk.csv")
test.head()
train_cpy=train.copy()
test_cpy=test.copy()
train.label.value_counts()
# Clean the tweets    


words_remove = ["ax","i","you","edu","s","t","m","subject","can","lines","re","what", "there","all","we",
                "one","the","a","an","of","or","in","for","by","on","but","is","in","a","not","with","as",
                "was","if","they","are","this","and","it","have","has","from","at","my","be","by","not","that",
                "to","from","com","org","so","said","from","what","told","over","more","other",
                "have","last","with","this","that","such","when","been","says","will","also","where","why",
                "would","today", "in", "on", "you", "r", "d", "u", "hw","wat", "oly", "s", "b", "ht", 
                "rt", "p","the","th", "n", "was"]


def cleantext(df, words_to_remove = words_remove): 
    
    df['cleaned_tweet'] = df['tweet'].replace(r'\'|\"|\,|\.|\?|\+|\-|\/|\=|\(|\)|\n|"', '', regex=True)
    df['cleaned_tweet'] = df['cleaned_tweet'].replace("  ", " ")

    ### dont change the original tweet
    # remove emoticons form the tweets
    df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'<ed>','', regex = True)
    df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'\B<U+.*>|<U+.*>\B|<U+.*>','', regex = True)
    
    # convert tweets to lowercase
    df['cleaned_tweet'] = df['cleaned_tweet'].str.lower()
    
    #remove user mentions
    df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'^(@\w+)',"", regex=True)
    
    #remove_symbols
    df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'[^a-zA-Z0-9]', " ", regex=True)

    #remove punctuations 
    df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+',"", regex = True)

    #remove_URL(x):
    df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'https.*$', "", regex = True)

    #remove 'amp' in the text
    df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'amp',"", regex = True)
    
    #remove words of length 1 or 2 
    df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'\b[a-zA-Z]{1,2}\b','', regex=True)

    #remove extra spaces in the tweet
    df['cleaned_tweet'] = df['cleaned_tweet'].replace(r'^\s+|\s+$'," ", regex=True)
     
    
    #remove stopwords and words_to_remove
    stop_words = set(stopwords.words('english'))
    mystopwords = [stop_words, "via", words_to_remove]
    
    df['fully_cleaned_tweet'] = df['cleaned_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in mystopwords]))
    

    return df

#get the processed tweets
train = cleantext(train,words_remove)
test = cleantext(test,words_remove)
train.head()
train['tokenized_tweet']=train['fully_cleaned_tweet'].apply(word_tokenize)
test['tokenized_tweet']=test['fully_cleaned_tweet'].apply(word_tokenize)
train.head()
test.head()
# if word has a digit, remove that word
train['tokenized_tweet']=train['tokenized_tweet'].apply(lambda x: [y for y in x if not any(c.isdigit() for c in y)])
# if word has a digit, remove that word
test['tokenized_tweet']=test['tokenized_tweet'].apply(lambda x: [y for y in x if not any(c.isdigit() for c in y)])
train.head()
stemmer = PorterStemmer()

train['tokenized_tweet'] = train['tokenized_tweet'].apply(lambda x: [stemmer.stem(i) for i in x])
test['tokenized_tweet'] = test['tokenized_tweet'].apply(lambda x: [stemmer.stem(i) for i in x])
train.head()
from sklearn.feature_extraction.text import CountVectorizer
bow_vec=CountVectorizer(max_df=0.7, min_df=2, max_features=1000, stop_words='english')
train['tokenized_tweet']=[" ".join(tokenized_tweet) for tokenized_tweet in train['tokenized_tweet'].values]
test['tokenized_tweet']=[" ".join(tokenized_tweet) for tokenized_tweet in test['tokenized_tweet'].values]
train.head()
train['count']=train['tokenized_tweet'].str.split().str.len()
test['count']=test['tokenized_tweet'].str.split().str.len()
train['count']=train['count'].astype('category')
test['count']=test['count'].astype('category')
y=train['label']
train.drop(['id','tweet','cleaned_tweet','fully_cleaned_tweet','label'],axis=1,inplace=True)
test.drop(['id','tweet','cleaned_tweet','fully_cleaned_tweet'],axis=1,inplace=True)
train_bow=bow_vec.fit_transform(train['tokenized_tweet'])
test_bow=bow_vec.transform(test['tokenized_tweet'])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec=TfidfVectorizer(max_df=0.7, min_df=2,max_features=1000, stop_words='english')
train_tfidf=tfidf_vec.fit_transform(train['tokenized_tweet'])
test_tfidf=tfidf_vec.transform(test['tokenized_tweet'])
### Understanding the common words

all_words = ' '.join([text for text in train['tokenized_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
normal_words =' '.join([text for text in train['tokenized_tweet'][y == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
negative_words =' '.join([text for text in train['tokenized_tweet'][y == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


X_train, X_test, y_train, y_test = train_test_split(train_bow, y, test_size=0.3, random_state=42)

lr=LogisticRegression(C=0.5)

lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
f1_score(y_test,y_pred)
y_test_pred=lr.predict(test_bow)
submission = pd.concat([test_cpy['id'],pd.DataFrame(y_test_pred, columns=['label'])],1)
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file
X_train, X_test, y_train, y_test = train_test_split(train_tfidf, y, test_size=0.3, random_state=42)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
f1_score(y_test,y_pred)
y_test_pred=lr.predict(test_tfidf)
submission = pd.concat([test_cpy['id'],pd.DataFrame(y_test_pred, columns=['label'])],1)
submission.to_csv('sub_lreg_tfidf.csv', index=False) # writing data to a CSV file
import lightgbm as lgb
lgbm=lgb.LGBMClassifier(reg_alpha=0.05)
lgbm.fit(X_train,y_train)
y_pred=lgbm.predict(X_test)
f1_score(y_test,y_pred)
y_test_pred=lgbm.predict(test_tfidf)
submission = pd.concat([test_cpy['id'],pd.DataFrame(y_test_pred, columns=['label'])],1)
submission.to_csv('sub_lgbm_tfidf.csv', index=False) # writing data to a CSV file
import catboost as cb
cb=cb.CatBoostClassifier()
cb.fit(X_train,y_train)
y_pred=cb.predict(X_test)
f1_score(y_test,y_pred)
y_test_pred=cb.predict(test_tfidf)
submission = pd.concat([test_cpy['id'],pd.DataFrame(y_test_pred, columns=['label'])],1)
submission.to_csv('sub_cb_tfidf.csv', index=False) # writing data to a CSV file
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
xgb=XGBClassifier()
params={'colsample_bytree': [0.7], 'learning_rate': [0.1], 'max_depth': [7], 'min_child_weight': [4],
        'n_estimators': [100],'objective': ['reg:logistic'], 'subsample': [0.6], 'verbosity': [1]}
from sklearn.model_selection import GridSearchCV
xgb1 = GridSearchCV(xgb,
                    params,
                    cv = 3,
                    n_jobs = 5,
                    verbose=True)

xgb1.fit(X_train,y_train)
y_pred=xgb1.predict(X_test)
f1_score(y_test,y_pred)
y_test_pred=xgb1.predict(test_tfidf)
submission = pd.concat([test_cpy['id'],pd.DataFrame(y_test_pred, columns=['label'])],1)
submission.to_csv('sub_xgb_tfidf.csv', index=False) # writing data to a CSV file
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
f1_score(y_test,y_pred)
y_test_pred=dt.predict(test_tfidf)
submission = pd.concat([test_cpy['id'],pd.DataFrame(y_test_pred, columns=['label'])],1)
submission.to_csv('sub_dt_tfidf.csv', index=False) # writing data to a CSV file