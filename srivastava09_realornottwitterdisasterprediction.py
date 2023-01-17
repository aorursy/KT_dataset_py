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
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import re

import numpy as np

from sklearn.model_selection import train_test_split

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords 

import nltk

import pdb as pdb

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.metrics import f1_score

import category_encoders as ce

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb



class DataPreprocessing:

	def __init__(self,filename):

		self.filename = filename

		self.tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

		self.labelencoder = LabelEncoder()

		self.ohencoder = OneHotEncoder()



	def preprocess(self):

		self.train_data = pd.read_csv(self.filename)

		input_data = self.train_data[['keyword', 'location', 'text']]

		output_data = self.train_data[['target']]





		hash_tags = input_data['text'].apply(lambda x : self.hashtag_extract(x))

		input_data['hash_tags'] = hash_tags

		#Remove all @user things from tweet -> they aren't of much use

		input_data['text'] = np.vectorize(self.remove_pattern)(input_data['text'],"@[\w]*")

		#Remove special chars

		input_data['text'] = input_data['text'].str.replace("[^a-zA-Z#]"," ")



		#Fill location with 'UNKNOWN'

		input_data['location'] = input_data['location'].fillna('UNKNOWN')



		#Fill keyword with 'UNAVAIALABLE'

		input_data['keyword'] = input_data['keyword'].fillna('UNAVAIALABLE')



		#Converting text to vector

		stop_words = set(stopwords.words('english'))

		tokenized_tweet = input_data['text'].apply(lambda x : x.split())

		stemmer = PorterStemmer()

		tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x if i not in stop_words])



		tokenized_tweet = tokenized_tweet.apply(lambda x: ' '.join(x))



		tokenized_tweet = self.tfidf_vectorizer.fit_transform(tokenized_tweet)



		#pdb.set_trace()

		

		columns_we_encoder = ['location']

		we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		location_label_encoder = we_encoder.fit_transform(input_data['location'],output_data['target']).add_suffix('_woe')

		columns_we_encoder = ['keyword']

		we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		keyword_label_encoder = we_encoder.fit_transform(input_data['keyword'],output_data['target']).add_suffix('_woe')







		#location_label_encoder = self.labelencoder.fit_transform(input_data['location']).reshape(-1,1)

		#keyword_label_encoder = self.labelencoder.fit_transform(input_data['keyword']).reshape(-1,1)

		



		#tweets

		X_train,X_valid,y_train,y_valid = train_test_split(tokenized_tweet,output_data)

		clf = linear_model.RidgeClassifier()

		bst = clf.fit(X_train, y_train["target"])



		#Categorical

		zipped_location_keyword_data = zip(location_label_encoder,keyword_label_encoder)

		zipped_location_keyword_data_list = [[data[0][0],data[1][0]] for data in zipped_location_keyword_data]





		X_numeric_train = np.array(zipped_location_keyword_data_list)[y_train.index]

		X_numeric_valid = np.array(zipped_location_keyword_data_list)[y_valid.index]



		clf_numeric = linear_model.RidgeClassifier()

		bst_numeric = clf_numeric.fit(X_numeric_train,y_train["target"])



		#Merging

		X_train_tweet_new = bst.decision_function(X_train)

		X_train_numeric_new = bst_numeric.decision_function(X_numeric_train)



		zipped_new_data = zip(X_train_tweet_new,X_train_numeric_new)

		zipped_new_data_list = [[data[0],data[0]] for data in zipped_new_data]

		zipped_new_data_list = np.array(zipped_new_data_list)



		clf_merged = linear_model.RidgeClassifier()

		bst_merged = clf_merged.fit(zipped_new_data_list,y_train["target"])



		#pdb.set_trace()





		#Validation

		output_tweets = bst.decision_function(X_valid)

		output_numerics = bst_numeric.decision_function(X_numeric_valid)



		zipped_output = zip(output_tweets,output_numerics)

		zipped_output_list = [[data[0],data[0]] for data in zipped_output]

		zipped_output_list = np.array(zipped_output_list)



		y_predict = bst_merged.predict(zipped_output_list)

		score = f1_score(y_valid['target'],y_predict)

		print(score)



	def preprocess_woe(self):

		self.train_data = pd.read_csv(self.filename)

		input_data = self.train_data[['keyword', 'location', 'text']]

		output_data = self.train_data[['target']]





		hash_tags = input_data['text'].apply(lambda x : self.hashtag_extract(x))

		input_data['hash_tags'] = hash_tags

		#Remove all @user things from tweet -> they aren't of much use

		input_data['text'] = np.vectorize(self.remove_pattern)(input_data['text'],"@[\w]*")

		#Remove special chars

		input_data['text'] = input_data['text'].str.replace("[^a-zA-Z#]"," ")



		#Fill location with 'UNKNOWN'

		input_data['location'] = input_data['location'].fillna('UNKNOWN')



		#Fill keyword with 'UNAVAIALABLE'

		input_data['keyword'] = input_data['keyword'].fillna('UNAVAIALABLE')



		#Converting text to vector

		stop_words = set(stopwords.words('english'))

		tokenized_tweet = input_data['text'].apply(lambda x : x.split())

		stemmer = PorterStemmer()

		tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x if i not in stop_words])



		tokenized_tweet = tokenized_tweet.apply(lambda x: ' '.join(x))



		tokenized_tweet = self.tfidf_vectorizer.fit_transform(tokenized_tweet)



		#pdb.set_trace()

		

		columns_we_encoder = ['location','keyword']

		we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		woe_label_encoder = we_encoder.fit_transform(input_data[columns_we_encoder],output_data['target']).add_suffix('_woe')

		#columns_we_encoder = ['keyword']

		#we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		#keyword_label_encoder = we_encoder.fit_transform(input_data['keyword'],output_data['target']).add_suffix('_woe')







		#location_label_encoder = self.labelencoder.fit_transform(input_data['location']).reshape(-1,1)

		#keyword_label_encoder = self.labelencoder.fit_transform(input_data['keyword']).reshape(-1,1)

		



		#tweets

		X_train,X_valid,y_train,y_valid = train_test_split(tokenized_tweet,output_data)

		clf = linear_model.RidgeClassifier()

		bst = clf.fit(X_train, y_train["target"])



		#Categorical

		#zipped_location_keyword_data = zip(location_label_encoder,keyword_label_encoder)

		#zipped_location_keyword_data_list = [[data[0][0],data[1][0]] for data in zipped_location_keyword_data]





		X_numeric_train = np.array(woe_label_encoder)[y_train.index]

		X_numeric_valid = np.array(woe_label_encoder)[y_valid.index]



		clf_numeric = linear_model.RidgeClassifier()

		bst_numeric = clf_numeric.fit(X_numeric_train,y_train["target"])



		#Merging

		X_train_tweet_new = bst.decision_function(X_train)

		X_train_numeric_new = bst_numeric.decision_function(X_numeric_train)



		zipped_new_data = zip(X_train_tweet_new,X_train_numeric_new)

		zipped_new_data_list = [[data[0],data[0]] for data in zipped_new_data]

		zipped_new_data_list = np.array(zipped_new_data_list)



		clf_merged = linear_model.RidgeClassifier()

		bst_merged = clf_merged.fit(zipped_new_data_list,y_train["target"])



		#Validation

		output_tweets = bst.decision_function(X_valid)

		output_numerics = bst_numeric.decision_function(X_numeric_valid)



		zipped_output = zip(output_tweets,output_numerics)

		zipped_output_list = [[data[0],data[0]] for data in zipped_output]

		zipped_output_list = np.array(zipped_output_list)



		y_predict = bst_merged.predict(zipped_output_list)

		score = f1_score(y_valid['target'],y_predict)

		print(score)

		'''

		#How to convert #Tags

		hash_tags_all = sum(hash_tags,[])

		unique_hashtags = pd.Series(hash_tags_all).str.lower().unique()

		unique_hashtags_int_tag = dict(enumerate(unique_hashtags))

		unique_hashtags_tag_int = {word : index for index,word in unique_hashtags_int_tag.items()}

		'''



	def preprocess_woe_svm(self):

		self.train_data = pd.read_csv(self.filename)

		input_data = self.train_data[['keyword', 'location', 'text']]

		output_data = self.train_data[['target']]





		hash_tags = input_data['text'].apply(lambda x : self.hashtag_extract(x))

		input_data['hash_tags'] = hash_tags

		#Remove all @user things from tweet -> they aren't of much use

		input_data['text'] = np.vectorize(self.remove_pattern)(input_data['text'],"@[\w]*")

		#Remove special chars

		input_data['text'] = input_data['text'].str.replace("[^a-zA-Z#]"," ")



		#Fill location with 'UNKNOWN'

		input_data['location'] = input_data['location'].fillna('UNKNOWN')



		#Fill keyword with 'UNAVAIALABLE'

		input_data['keyword'] = input_data['keyword'].fillna('UNAVAIALABLE')



		#Converting text to vector

		stop_words = set(stopwords.words('english'))

		tokenized_tweet = input_data['text'].apply(lambda x : x.split())

		stemmer = PorterStemmer()

		tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x if i not in stop_words])



		tokenized_tweet = tokenized_tweet.apply(lambda x: ' '.join(x))



		tokenized_tweet = self.tfidf_vectorizer.fit_transform(tokenized_tweet)

		

		columns_we_encoder = ['location','keyword']

		we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		woe_label_encoder = we_encoder.fit_transform(input_data[columns_we_encoder],output_data['target']).add_suffix('_woe')

		#columns_we_encoder = ['keyword']

		#we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		#keyword_label_encoder = we_encoder.fit_transform(input_data['keyword'],output_data['target']).add_suffix('_woe')







		#location_label_encoder = self.labelencoder.fit_transform(input_data['location']).reshape(-1,1)

		#keyword_label_encoder = self.labelencoder.fit_transform(input_data['keyword']).reshape(-1,1)

		



		#tweets

		X_train,X_valid,y_train,y_valid = train_test_split(tokenized_tweet,output_data)

		clf = linear_model.RidgeClassifier()

		bst = clf.fit(X_train, y_train["target"])



		#Categorical

		#zipped_location_keyword_data = zip(location_label_encoder,keyword_label_encoder)

		#zipped_location_keyword_data_list = [[data[0][0],data[1][0]] for data in zipped_location_keyword_data]





		X_numeric_train = np.array(woe_label_encoder)[y_train.index]

		X_numeric_valid = np.array(woe_label_encoder)[y_valid.index]





		clf_numeric = svm.SVC(kernel = 'rbf')

		bst_numeric = clf_numeric.fit(X_numeric_train,y_train["target"])



		#Merging

		X_train_tweet_new = bst.decision_function(X_train)

		X_train_numeric_new = bst_numeric.decision_function(X_numeric_train)



		zipped_new_data = zip(X_train_tweet_new,X_train_numeric_new)

		zipped_new_data_list = [[data[0],data[0]] for data in zipped_new_data]

		zipped_new_data_list = np.array(zipped_new_data_list)



		clf_merged = linear_model.RidgeClassifier()

		bst_merged = clf_merged.fit(zipped_new_data_list,y_train["target"])



		#Validation

		output_tweets = bst.decision_function(X_valid)

		output_numerics = bst_numeric.decision_function(X_numeric_valid)



		zipped_output = zip(output_tweets,output_numerics)

		zipped_output_list = [[data[0],data[0]] for data in zipped_output]

		zipped_output_list = np.array(zipped_output_list)



		y_predict = bst_merged.predict(zipped_output_list)

		score = f1_score(y_valid['target'],y_predict)

		print(score)



		#pdb.set_trace()

		test = pd.read_csv('test.csv',index_col = 'id')

		input_data = test[['keyword', 'location', 'text']]

		input_data['text'] = np.vectorize(self.remove_pattern)(input_data['text'],"@[\w]*")

		input_data['text'] = input_data['text'].str.replace("[^a-zA-Z#]"," ")

		input_data['location'] = input_data['location'].fillna('UNKNOWN')

		input_data['keyword'] = input_data['keyword'].fillna('UNAVAIALABLE')

		tokenized_tweet = input_data['text'].apply(lambda x : x.split())

		tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x if i not in stop_words])

		tokenized_tweet = tokenized_tweet.apply(lambda x: ' '.join(x))

		tokenized_tweet = self.tfidf_vectorizer.transform(tokenized_tweet)

		woe_label_encoder = we_encoder.transform(input_data[columns_we_encoder]).add_suffix('_woe')



		tweet = bst.decision_function(tokenized_tweet)

		numeric = bst_numeric.decision_function(np.array(woe_label_encoder))



		zipped_output = zip(tweet,numeric)

		zipped_output_list = [[data[0],data[0]] for data in zipped_output]

		zipped_output_list = np.array(zipped_output_list)

		y_predict = bst_merged.predict(zipped_output_list)

		test['target'] = y_predict

		test['target'].to_csv('submission_n.csv')



		'''

		#How to convert #Tags

		hash_tags_all = sum(hash_tags,[])

		unique_hashtags = pd.Series(hash_tags_all).str.lower().unique()

		unique_hashtags_int_tag = dict(enumerate(unique_hashtags))

		unique_hashtags_tag_int = {word : index for index,word in unique_hashtags_int_tag.items()}

		'''

	def preprocess_woe_decision_tree(self):

		self.train_data = pd.read_csv(self.filename)

		input_data = self.train_data[['keyword', 'location', 'text']]

		output_data = self.train_data[['target']]





		hash_tags = input_data['text'].apply(lambda x : self.hashtag_extract(x))

		input_data['hash_tags'] = hash_tags

		#Remove all @user things from tweet -> they aren't of much use

		input_data['text'] = np.vectorize(self.remove_pattern)(input_data['text'],"@[\w]*")

		#Remove special chars

		input_data['text'] = input_data['text'].str.replace("[^a-zA-Z#]"," ")



		#Fill location with 'UNKNOWN'

		input_data['location'] = input_data['location'].fillna('UNKNOWN')



		#Fill keyword with 'UNAVAIALABLE'

		input_data['keyword'] = input_data['keyword'].fillna('UNAVAIALABLE')



		#Converting text to vector

		stop_words = set(stopwords.words('english'))

		tokenized_tweet = input_data['text'].apply(lambda x : x.split())

		stemmer = PorterStemmer()

		tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x if i not in stop_words])



		tokenized_tweet = tokenized_tweet.apply(lambda x: ' '.join(x))



		tokenized_tweet = self.tfidf_vectorizer.fit_transform(tokenized_tweet)

		

		columns_we_encoder = ['location','keyword']

		we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		woe_label_encoder = we_encoder.fit_transform(input_data[columns_we_encoder],output_data['target']).add_suffix('_woe')

		#columns_we_encoder = ['keyword']

		#we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		#keyword_label_encoder = we_encoder.fit_transform(input_data['keyword'],output_data['target']).add_suffix('_woe')







		#location_label_encoder = self.labelencoder.fit_transform(input_data['location']).reshape(-1,1)

		#keyword_label_encoder = self.labelencoder.fit_transform(input_data['keyword']).reshape(-1,1)

		



		#tweets

		X_train,X_valid,y_train,y_valid = train_test_split(tokenized_tweet,output_data)

		clf = linear_model.RidgeClassifier()

		bst = clf.fit(X_train, y_train["target"])



		#Categorical

		#zipped_location_keyword_data = zip(location_label_encoder,keyword_label_encoder)

		#zipped_location_keyword_data_list = [[data[0][0],data[1][0]] for data in zipped_location_keyword_data]





		X_numeric_train = np.array(woe_label_encoder)[y_train.index]

		X_numeric_valid = np.array(woe_label_encoder)[y_valid.index]



		#pdb.set_trace()





		clf_numeric = DecisionTreeClassifier()

		bst_numeric = clf_numeric.fit(X_numeric_train,y_train["target"])



		#Merging

		X_train_tweet_new = bst.decision_function(X_train)

		X_train_numeric_new = bst_numeric.predict_proba(X_numeric_train)



		zipped_new_data = zip(X_train_tweet_new,X_train_numeric_new[:,1])

		zipped_new_data_list = [[data[0],data[0]] for data in zipped_new_data]

		zipped_new_data_list = np.array(zipped_new_data_list)



		clf_merged = linear_model.RidgeClassifier()

		bst_merged = clf_merged.fit(zipped_new_data_list,y_train["target"])



		#Validation

		output_tweets = bst.decision_function(X_valid)

		output_numerics = bst_numeric.predict_proba(X_numeric_valid)



		zipped_output = zip(output_tweets,output_numerics[:,1])

		zipped_output_list = [[data[0],data[0]] for data in zipped_output]

		zipped_output_list = np.array(zipped_output_list)



		y_predict = bst_merged.predict(zipped_output_list)

		score = f1_score(y_valid['target'],y_predict)

		print(score)



		#pdb.set_trace()

		test = pd.read_csv('test.csv',index_col = 'id')

		input_data = test[['keyword', 'location', 'text']]

		input_data['text'] = np.vectorize(self.remove_pattern)(input_data['text'],"@[\w]*")

		input_data['text'] = input_data['text'].str.replace("[^a-zA-Z#]"," ")

		input_data['location'] = input_data['location'].fillna('UNKNOWN')

		input_data['keyword'] = input_data['keyword'].fillna('UNAVAIALABLE')

		tokenized_tweet = input_data['text'].apply(lambda x : x.split())

		tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x if i not in stop_words])

		tokenized_tweet = tokenized_tweet.apply(lambda x: ' '.join(x))

		tokenized_tweet = self.tfidf_vectorizer.transform(tokenized_tweet)

		woe_label_encoder = we_encoder.transform(input_data[columns_we_encoder]).add_suffix('_woe')



		tweet = bst.decision_function(tokenized_tweet)

		numeric = bst_numeric.predict_proba(np.array(woe_label_encoder))



		zipped_output = zip(tweet,numeric[:,1])

		zipped_output_list = [[data[0],data[0]] for data in zipped_output]

		zipped_output_list = np.array(zipped_output_list)

		y_predict = bst_merged.predict(zipped_output_list)

		test['target'] = y_predict

		test['target'].to_csv('submission_n.csv')



		'''

		#How to convert #Tags

		hash_tags_all = sum(hash_tags,[])

		unique_hashtags = pd.Series(hash_tags_all).str.lower().unique()

		unique_hashtags_int_tag = dict(enumerate(unique_hashtags))

		unique_hashtags_tag_int = {word : index for index,word in unique_hashtags_int_tag.items()}

		'''



	def preprocess_woe_xgboost(self):

		#pdb.set_trace()

		self.train_data = pd.read_csv(self.filename)

		input_data = self.train_data[['keyword', 'location', 'text']]

		output_data = self.train_data[['target']]





		hash_tags = input_data['text'].apply(lambda x : self.hashtag_extract(x))

		input_data['hash_tags'] = hash_tags

		#Remove all @user things from tweet -> they aren't of much use

		input_data['text'] = np.vectorize(self.remove_pattern)(input_data['text'],"@[\w]*")

		#Remove special chars

		input_data['text'] = input_data['text'].str.replace("[^a-zA-Z#]"," ")



		#Fill location with 'UNKNOWN'

		input_data['location'] = input_data['location'].fillna('UNKNOWN')



		#Fill keyword with 'UNAVAIALABLE'

		input_data['keyword'] = input_data['keyword'].fillna('UNAVAIALABLE')



		#Converting text to vector

		stop_words = set(stopwords.words('english'))

		tokenized_tweet = input_data['text'].apply(lambda x : x.split())

		stemmer = PorterStemmer()

		tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x if i not in stop_words])



		tokenized_tweet = tokenized_tweet.apply(lambda x: ' '.join(x))



		tokenized_tweet = self.tfidf_vectorizer.fit_transform(tokenized_tweet)

		

		columns_we_encoder = ['location','keyword']

		we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		woe_label_encoder = we_encoder.fit_transform(input_data[columns_we_encoder],output_data['target']).add_suffix('_woe')

		#columns_we_encoder = ['keyword']

		#we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		#keyword_label_encoder = we_encoder.fit_transform(input_data['keyword'],output_data['target']).add_suffix('_woe')







		#location_label_encoder = self.labelencoder.fit_transform(input_data['location']).reshape(-1,1)

		#keyword_label_encoder = self.labelencoder.fit_transform(input_data['keyword']).reshape(-1,1)

		



		#tweets

		X_train,X_valid,y_train,y_valid = train_test_split(tokenized_tweet,output_data)

		clf = linear_model.RidgeClassifier()

		bst = clf.fit(X_train, y_train["target"])



		#Categorical

		#zipped_location_keyword_data = zip(location_label_encoder,keyword_label_encoder)

		#zipped_location_keyword_data_list = [[data[0][0],data[1][0]] for data in zipped_location_keyword_data]





		X_numeric_train = np.array(woe_label_encoder)[y_train.index]

		X_numeric_valid = np.array(woe_label_encoder)[y_valid.index]





		clf_numeric = xgb.XGBRegressor(objective="binary:logistic",random_state=42)

		bst_numeric = clf_numeric.fit(X_numeric_train,y_train["target"])



		#Merging

		X_train_tweet_new = bst.decision_function(X_train)

		X_train_numeric_new = bst_numeric.predict(X_numeric_train)



		zipped_new_data = zip(X_train_tweet_new,X_train_numeric_new)

		zipped_new_data_list = [[data[0],data[0]] for data in zipped_new_data]

		zipped_new_data_list = np.array(zipped_new_data_list)



		clf_merged = linear_model.RidgeClassifier()

		bst_merged = clf_merged.fit(zipped_new_data_list,y_train["target"])



		#pdb.set_trace()



		#Validation

		output_tweets = bst.decision_function(X_valid)

		output_numerics = bst_numeric.predict(X_numeric_valid)



		zipped_output = zip(output_tweets,output_numerics)

		zipped_output_list = [[data[0],data[0]] for data in zipped_output]

		zipped_output_list = np.array(zipped_output_list)



		y_predict = bst_merged.predict(zipped_output_list)

		score = f1_score(y_valid['target'],y_predict)

		print(score)



		#pdb.set_trace()

		test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv',index_col = 'id')

		input_data = test[['keyword', 'location', 'text']]

		input_data['text'] = np.vectorize(self.remove_pattern)(input_data['text'],"@[\w]*")

		input_data['text'] = input_data['text'].str.replace("[^a-zA-Z#]"," ")

		input_data['location'] = input_data['location'].fillna('UNKNOWN')

		input_data['keyword'] = input_data['keyword'].fillna('UNAVAIALABLE')

		tokenized_tweet = input_data['text'].apply(lambda x : x.split())

		tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x if i not in stop_words])

		tokenized_tweet = tokenized_tweet.apply(lambda x: ' '.join(x))

		tokenized_tweet = self.tfidf_vectorizer.transform(tokenized_tweet)

		woe_label_encoder = we_encoder.transform(input_data[columns_we_encoder]).add_suffix('_woe')



		tweet = bst.decision_function(tokenized_tweet)

		numeric = bst_numeric.predict(np.array(woe_label_encoder))



		zipped_output = zip(tweet,numeric)

		zipped_output_list = [[data[0],data[0]] for data in zipped_output]

		zipped_output_list = np.array(zipped_output_list)

		y_predict = bst_merged.predict(zipped_output_list)

		test['target'] = y_predict

		test['target'].to_csv('submission.csv')

		'''

		#How to convert #Tags

		hash_tags_all = sum(hash_tags,[])

		unique_hashtags = pd.Series(hash_tags_all).str.lower().unique()

		unique_hashtags_int_tag = dict(enumerate(unique_hashtags))

		unique_hashtags_tag_int = {word : index for index,word in unique_hashtags_int_tag.items()}

		'''



	def train_test_split(self,input_data,output_data):

		#X_zipped_list = [data for data in list(zip(input_data__text_tfidf,input_data__location,input_data__keywords))]

		return train_test_split(input_data,output_data,test_size=0.33,random_state=42)



	def hashtag_extract(self,x):

		ht = re.findall(r"#(\w+)",x)

		if len(ht) == 0:

			ht = ['empty']

		return ht



	def remove_pattern(self,input_txt,pattern):

		r = re.findall(pattern,input_txt)

		for i in r:

			input_txt = re.sub(i,'',input_txt)

		return input_txt



	def create_submission_file(self,filename,model1,model2,model3):

		#pdb.set_trace()

		test = pd.read_csv(filename,index_col = 'id')

		input_data = test[['keyword', 'location', 'text']]

		input_data['text'] = np.vectorize(self.remove_pattern)(input_data['text'],"@[\w]*")

		input_data['text'] = input_data['text'].str.replace("[^a-zA-Z#]"," ")

		input_data['location'] = input_data['location'].fillna('UNKNOWN')

		input_data['keyword'] = input_data['keyword'].fillna('UNAVAIALABLE')

		stop_words = set(stopwords.words('english'))

		tokenized_tweet = input_data['text'].apply(lambda x : x.split())

		stemmer = PorterStemmer()

		tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x if i not in stop_words])



		tokenized_tweet = tokenized_tweet.apply(lambda x: ' '.join(x))



		tokenized_tweet = self.tfidf_vectorizer.fit_transform(tokenized_tweet)

		

		columns_we_encoder = ['location','keyword']

		we_encoder = ce.WOEEncoder(cols = columns_we_encoder)

		woe_label_encoder = we_encoder.fit_transform(input_data[columns_we_encoder],output_data['target']).add_suffix('_woe')



		tweet = model1.decision_function(tokenized_tweet)

		numeric = model2.predict(np.array(woe_label_encoder))



		zipped_output = zip(tweet,numeric)

		zipped_output_list = [[data[0],data[0]] for data in zipped_output]

		zipped_output_list = np.array(zipped_output_list)

		y_predict = bst_merged.predict(zipped_output_list)

		test['target'] = y_predict







A = DataPreprocessing('/kaggle/input/nlp-getting-started/train.csv')

A.preprocess_woe_xgboost()

#A.create_submission_file('test.csv',model1,model2,model3)
