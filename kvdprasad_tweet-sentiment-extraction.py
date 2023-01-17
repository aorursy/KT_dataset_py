# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing,metrics

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #### 1.	Load the dataset and create a dataframe.
train_reviews_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_reviews_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
submission_sample_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

train_reviews_df
train_reviews_df
test_reviews_df
train_tweets_df = train_reviews_df[['text','sentiment']]
train_tweets_df
test_tweets_df = test_reviews_df[['text','sentiment']]
test_tweets_df
train_tweets_df.shape, test_tweets_df.shape
train_tweets_df.isna().sum()
test_tweets_df.isna().sum()
train_tweets_df = train_tweets_df.dropna()
train_tweets_df.shape
#Preprocessing
from nltk.corpus import stopwords
def tweet_preprocessing(tweets_df):
    
    # Replace special symbols
    tweets_df['processed_text'] = tweets_df['text'].replace(to_replace ='(@[\w]+)', value ='', regex = True) 
    
    #remove any links from the tweet: Links not required for performing sentiment analysis.
    tweets_df['processed_text'] = tweets_df['processed_text'].str.replace('((www\.[\s]+)|(https?://[^\s]+))','\0',regex=True)
    
    # remove special characters, numbers, punctuations: None of them would add any value to the sentiment score.
    tweets_df['processed_text'] = tweets_df['processed_text'].str.replace("[^a-zA-Z]+", " ")
    
    #Converting into lower case and splitting into wrods.
    tweets_df["processed_text"] = tweets_df["processed_text"].str.lower()
    tweets_df["processed_text"] = tweets_df["processed_text"].str.split()
    
    stop = stopwords.words('english')
    #tweets_df['processed_text']=tweets_df['processed_text'].apply(lambda x: [item for item in x if item not in stop])

    return tweets_df
import warnings

warnings.filterwarnings("ignore")

train_tweets_df = tweet_preprocessing(train_tweets_df)
train_tweets_df.head()
test_tweets_df = tweet_preprocessing(test_tweets_df)
test_tweets_df.head()
def rejoin_words(train_tweets_df):
    my_list = train_tweets_df['processed_text']
    joined_words = ( " ".join(my_list))
    #print(joined_words)
    return joined_words

train_tweets_df['new_processed_text'] = train_tweets_df.apply(rejoin_words, axis=1)
train_tweets_df
test_tweets_df['new_processed_text'] = test_tweets_df.apply(rejoin_words, axis=1)
test_tweets_df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vectorizer = CountVectorizer(max_features=23618)
test_x = test_tweets_df['new_processed_text']
test_transformed_vector = count_vectorizer.fit_transform(test_x)
test_transformed_vector.shape
test_transformed_vector.shape[1]
count_vectorizer = CountVectorizer(max_features = 6770)
X = train_tweets_df['new_processed_text']
train_transformed_vector = count_vectorizer.fit_transform(X)
train_transformed_vector.shape
from sklearn.model_selection import train_test_split
Y = train_tweets_df['sentiment']
x_train, x_test, y_train, y_test = train_test_split(train_transformed_vector, Y, test_size = 0.2)
x_train.shape, x_test.shape
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(x_train.toarray(), y_train)
GNB_pred=clf.predict(x_test.toarray())

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
model_perforamnce = confusion_matrix(y_test,GNB_pred)
model_perforamnce
print (classification_report(y_test, GNB_pred))
# Let's do the import for Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# creating instance
tweet_analysis_mnb = MultinomialNB()
tweet_analysis_mnb.fit(x_train, y_train)
mnb_pred = tweet_analysis_mnb.predict(x_test)
from sklearn.metrics import confusion_matrix
MNB_model_perforamnce = confusion_matrix(y_test,mnb_pred)
MNB_model_perforamnce
mnb_pred
print (classification_report(y_test,mnb_pred))
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
lr_prediction = logistic_regression.predict(x_test)
from sklearn.metrics import confusion_matrix
lr_model_perforamnce = confusion_matrix(y_test,lr_prediction)
lr_model_perforamnce
print (classification_report(y_test,lr_prediction))
from sklearn.ensemble import RandomForestClassifier
random_forest_classifer = RandomForestClassifier()
random_forest_classifer.fit(x_train, y_train)
rfc_prediction = random_forest_classifer.predict(x_test)
from sklearn.metrics import confusion_matrix
rfc_model_perforamnce = confusion_matrix(y_test,rfc_prediction)
rfc_model_perforamnce

rfc_classification_report = classification_report(y_test,rfc_prediction)
print (rfc_classification_report)
from sklearn import svm
#Create a svm Classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(x_train, y_train)# predict the labels on validation dataset
predictions_SVM = SVM.predict(x_test)# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)

rfc_classification_report = classification_report(y_test,predictions_SVM)
print (rfc_classification_report)
def prepare_test_prediction_report(y_test, GNB_pred, mnb_pred, lr_prediction, rfc_prediction, predictions_SVM):
    frame = { 'Actual Value': y_test,'SVM':predictions_SVM,'GNB ': GNB_pred, 'MNB' :mnb_pred, 'LR':lr_prediction,'RFC':rfc_prediction} 
    result_pd = pd.DataFrame(frame)
    return result_pd
result_pd = prepare_test_prediction_report(y_test, GNB_pred, mnb_pred, lr_prediction, rfc_prediction, predictions_SVM)
result_pd
test_data_svm_pred = SVM.predict(test_transformed_vector.toarray())
test_data_svm_pred

# SVM for multi-class classification using built-in one-vs-one
from sklearn.datasets import make_classification
from sklearn.svm import SVC
# define dataset
# define model
svc = SVC(decision_function_shape='ovo')
# fit model
#train_tweets_df['sentiment']
#Y = pd.get_dummies(train_tweets_df.sentiment)
train_tweets_df['sentiment'].replace(to_replace=['negative', 'neutral', 'positive'], value=[1, 2, 3], inplace=True)
Y = train_tweets_df['sentiment']
svc.fit(train_transformed_vector, Y)
# make predictions

svc_pred = svc.predict(train_transformed_vector)

svc_classification_report = classification_report(Y,svc_pred)
print (svc_classification_report)
    
def prediction_report():
    test_data_svm_pred = SVM.predict(test_transformed_vector.toarray())
    test_reviews_df['SVM'] = test_data_svm_pred
    svc_test_pred = svc.predict(test_transformed_vector)
    test_reviews_df['SVC'] = svc_test_pred
    test_reviews_df['SVC'].replace(to_replace=[1, 2, 3], value=['negative', 'neutral', 'positive'], inplace=True)
    lr_prediction = logistic_regression.predict(test_transformed_vector)
    test_reviews_df['LR'] = lr_prediction
    rfc_prediction = random_forest_classifer.predict(test_transformed_vector)
    test_reviews_df['RFC'] = rfc_prediction
    mnb_pred = tweet_analysis_mnb.predict(test_transformed_vector)
    test_reviews_df['MNB'] = mnb_pred
    GNB_pred=clf.predict(test_transformed_vector.toarray())
    test_reviews_df['GNB'] = GNB_pred
    return test_reviews_df
prediction_report()
from sklearn.metrics import precision_recall_fscore_support as score
precision,recall,fscore,support=score(Y,svc_pred,average='macro')
print('Precision : {}'.format(precision*100)) 
print( 'Recall    : {}'.format(recall*100))
print( 'F-score   : {}'.format(fscore*100))
