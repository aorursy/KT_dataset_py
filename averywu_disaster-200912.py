import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# import nltk
import sklearn
from sklearn import feature_extraction, model_selection, linear_model
import re
###Check training data###

##Initial check##
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_train.shape
# (7613, 5)
df_train.columns
# Index(['id', 'keyword', 'location', 'text', 'target'], dtype='object')
df_train
##Find out how many unique elements are in each column##

kw_set = set(df_train.keyword)
loc_set = set(df_train.location)
text_set = set(df_train.text)

summary = pd.DataFrame(data = [[len(kw_set), len(loc_set), len(text_set)]], \
                       columns=["kw_set_len", "loc_set_len", "text_set_len"], \
                       index=["num_distinct_element"])
summary
##Compare positive data (i.e. disaster tweets) and negative data##
pos_vs_neg = df_train.groupby("target").count()
pos_vs_neg
##Compare positive data (i.e. disaster tweets) and negative data##
df_train_add = df_train.copy()
# df_train_add["text_len"] = df_train.text.apply(len) #WRONG: This count the number of characters!
df_train_add["text_len"] = df_train.text.apply(lambda x: len(x.split(" ")) if type(x)==str else x)
df_train_add.groupby("text_len").count()
# text_len: 1-54

df_train_add["keyword_len"] = df_train.keyword.apply(lambda x: len(x.split(" ")) if type(x)==str else x)
df_train_add.groupby("keyword_len").count()
#Keyword column: one or zero keyword
#keyword length: nah or 1

df_train_add["location_len"] = df_train.location.apply(lambda x: len(x.split(" ")) if type(x)==str else x)
df_train_add.groupby("location_len").count()
#Location column: nah or 1-11

# df_train_add.to_csv("df_train_add.csv")
pos_vs_neg_add = df_train_add.groupby("target").mean()
df_train.target==1

###Check testinging data###

df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_test.shape
# (3263, 4)
df_test.columns
# Index(['id', 'keyword', 'location', 'text'], dtype='object')
df_test
###Experiment with different libraries###

##Prepare examples

sample_text = df_train.text[:2]
for i, text in enumerate(sample_text):
    print(f"#{i} (word count: {len(text.split())}):\n{text}")
    print("-"*100)
##sklearn##

count_vectorizer = feature_extraction.text.CountVectorizer()
count_vectorizer
# CountVectorizer()

count_vectorizer.fit(sample_text) #The the text.CountVectorizer.fit method returnd ITSELF!
len(count_vectorizer.vocabulary_) #21637 when using all df_train's text data
count_vectorizer.fixed_vocabulary_
# False
count_vectorizer.stop_words_
# set()
vocab = count_vectorizer.vocabulary_
print(f"vocab size: {len(vocab)}")
sorted(vocab.items())
transformed = count_vectorizer.transform(sample_text)
# print(f"transformed.shape: {transformed.shape}")
for i in range(len(sample_text)):
    print(f"transformed[i].shape: {transformed[i].shape}")
    print(transformed[i])
# #     print("~"* 100)
# #     print(f"transformed[i].shape: {transformed[i].shape}")
# #     for j in range(transformed[i].shape[0]):
# #         print(transformed[i][j])
# #         print("-"*30)
        
    print("="*100)

transformed.toarray()
# array([[1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1],
#        [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]])
# print(sample_vec[0].todense().shape)
# print(sample_vec[0].todense())
###Represent each tweet as a vector, dimensions of which are the vocabulary of all word###

##sklearn.feature_extraction.text.CountVectorizer()##

#Build the dimensions (aka the vocabulary) using text data from df_train
count_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
count_vectorizer.fit(df_train.text)

count_vectorizer.fixed_vocabulary_
# False
count_vectorizer.stop_words_
# set()
count_vectorizer.vocabulary_
#Represent training_data as vectors
train_vectors = count_vectorizer.transform(df_train.text)
train_vectors
# <7613x21637 sparse matrix of type '<class 'numpy.int64'>'
# 	with 111497 stored elements in Compressed Sparse Row format>
# This means there are 7613 tweets, a vocabulary of 21637 words, and a total of 111497 words from these tweets.

test_vectors = count_vectorizer.transform(df_test.text)
##sklearn.feature_extraction.text.TfidfVectorizer()##

#Build the dimensions (aka the vocabulary) using text data from df_train
tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
tfidf_vectorizer.fit(df_train.text)

tfidf_vectorizer.fixed_vocabulary_
# False
tfidf_vectorizer.stop_words_
# set()
tfidf_vectorizer.vocabulary_
#Represent training_data as vectors
train_vectors_tfidf = tfidf_vectorizer.transform(df_train.text)
train_vectors_tfidf
# <7613x21637 sparse matrix of type '<class 'numpy.int64'>'
# 	with 111497 stored elements in Compressed Sparse Row format>
# This means there are 7613 tweets, a vocabulary of 21637 words, and a total of 111497 words from these tweets.
train_vectors_tfidf[0].todense()
###Build a model to analyze data###

ridge_clf = sklearn.linear_model.RidgeClassifier()
# ridge_clf.fit(train_vectors, df_train.target)
ridge_clf
count_vectorizer_scores = \
sklearn.model_selection.cross_val_score(ridge_clf, train_vectors, df_train.target, cv=3, scoring="f1")
count_vectorizer_scores
# array([0.59485531, 0.56526006, 0.64082434])

# tfidf_vectorizer_scores = \
# sklearn.model_selection.cross_val_score(ridge_clf, train_vectors_tfidf,df_train.target, cv=3, scoring="f1")
# tfidf_vectorizer_scores
# # array([0.63366337, 0.6122449 , 0.68442211])
text2 = "This is a book."
text3 = text2.replace("b", "B")
text4 = re.sub("a", "A", text2)
text3
text4
def clean_text(text):
    if type(text)==str:
        text = text.lower()
        text = re.sub(r"http\S+", "", text) 
    else:
        text = ""
    
    return text

def clean_data(df, cols):
    for col in cols:
#         df[col] = df[col].apply(lambda x:x.lower() if type(x)==str else "")
        df[col] = df[col].apply(clean_text)
    
    return df

df_train_clean = clean_data(df_train, ["keyword", "location", "text"])
df_train_clean
df_train.iloc[7610,:]["text"]
def judge_disaster(df_train, df_test, text_col, target_col, vectorizer):
    
    #Pick vectorizer
    count_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    vectorizer_dict = {"count_vectorizer": count_vectorizer,
                       "tfidf_vectorizer": tfidf_vectorizer}
    best_vectorizer = vectorizer_dict[vectorizer]
    
    #Pick classifier
    ridge_clf = sklearn.linear_model.RidgeClassifier()
    best_clf = ridge_clf
    
    #Vectorize tweets
    best_vectorizer.fit(df_train[text_col])
    train_vectors = best_vectorizer.transform(df_train[text_col])
    test_vectors = best_vectorizer.transform(df_test[text_col])
    
    #Cross-validation on df_train
    scores = \
    sklearn.model_selection.cross_val_score(best_clf, \
                                            train_vectors, \
                                            df_train[target_col], \
                                            cv=5, \
                                            scoring="f1")
    ave_score = sum(scores) / len(scores)
    
    #Use classifier to predict
    best_clf.fit(train_vectors, df_train[target_col])
    prediction = best_clf.predict(test_vectors)
    
    return ave_score, prediction

# judge1 = judge_disaster(df_train, df_test, "text", "target")
# set(judge1 == ridge_clf.predict(test_vectors))

ave_score, prediction = judge_disaster(df_train, df_test, "text", "target", "tfidf_vectorizer")
print(f"{ave_score}")
print(prediction)

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission

#Predict using RidgeClassfier
ridge_clf.fit(train_vectors, df_train.target) #Why does the classifier have to fit the data again???
sample_submission.target = ridge_clf.predict(test_vectors)
sample_submission.to_csv("submission_200912_count_vectorizer.csv")
sample_submission
