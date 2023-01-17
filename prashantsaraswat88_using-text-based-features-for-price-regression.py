import os

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import datetime

import math

import statistics as stats

import scipy

import seaborn
data_raw = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data_raw.head()
print("Dimensions: ", data_raw.shape)

num_row = data_raw.shape[0]

num_col = data_raw.shape[1]

print(data_raw.dtypes)
data = data_raw.copy() # Make new copy for cleaned data
data = data.drop(columns=['host_name']) # Not needed thanks to host_id. Anonymizes the data.
#Number of NaN values

print(data_raw.isna().sum())

print("Out of",num_row,"rows")
data['name'] = data['name'].fillna('')

data['reviews_per_month'] = data['reviews_per_month'].fillna(0)
data['last_review_DT'] = pd.to_datetime(data['last_review'])



#Fill 

mean_date = data['last_review_DT'].mean()

max_date = data['last_review_DT'].max()

data['last_review_DT'] = data['last_review_DT'].fillna(mean_date)



def how_many_days_ago(datetime):

    return (max_date - datetime).days



data['days_since_last_review'] = data['last_review_DT'].apply(how_many_days_ago)
from sklearn import preprocessing

from sklearn.compose import ColumnTransformer



# Here are the categorical features we are going to create one-hot encoded features for

categorical_features = ['neighbourhood_group','room_type','neighbourhood'] 



encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

one_hot_features = encoder.fit_transform(data[categorical_features])

one_hot_names = encoder.get_feature_names()



print("Type of one_hot_columns is:",type(one_hot_features))
one_hot_df = pd.DataFrame.sparse.from_spmatrix(one_hot_features)

one_hot_df.columns = one_hot_names # Now we can see the actual meaning of the one-hot feature in the DataFrame

one_hot_df.head()
import seaborn as sns

from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()



numerical_features = ['latitude','longitude','price','minimum_nights','number_of_reviews','reviews_per_month',

                      'days_since_last_review', 'calculated_host_listings_count','availability_365']



dataScaled = pd.DataFrame(min_max_scaler.fit_transform(data[numerical_features]), columns=numerical_features)

#viz_2=sns.violinplot(data=data, y=['price','number_of_reviews'])

ax = sns.boxplot(data=dataScaled, orient="h")

ax.set_title("Box plots for min-max scaled features")
sns.distplot(data['price']).set_title("Distribution of AirBnB prices")
# I'll transform the following columns by taking log(1+x)

transform_cols = ['price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count']

for col in transform_cols:

    col_log1p = col + '_log1p'

    data[col_log1p] = data[col].apply(math.log1p)
min_max_scaler = preprocessing.MinMaxScaler()



# Now let's plot the numerical features, but take the transformed values for the columns we applied log1p to

numerical_features_log1p = numerical_features

def take_log_col(col):

    if col in transform_cols: return col + '_log1p'

    else: return col

numerical_features_log1p[:] = [take_log_col(col) for col in numerical_features_log1p]



dataScaled_log1p = pd.DataFrame(min_max_scaler.fit_transform(data[numerical_features_log1p]), columns=numerical_features_log1p)

ax = sns.boxplot(data=dataScaled_log1p, orient="h")

ax.set_title("Box plots for min-max scaled features")
sns.distplot(data['price_log1p']).set_title("Distribution of log(1 + price)")
from sklearn.model_selection import train_test_split



numerical_feature_names = ['latitude', 'longitude', 'minimum_nights_log1p','number_of_reviews_log1p','reviews_per_month_log1p', 

                          'days_since_last_review', 'calculated_host_listings_count_log1p', 'availability_365']

numerical_features = data[numerical_feature_names]

scaler = preprocessing.MinMaxScaler()

numerical_features = scaler.fit_transform(numerical_features) # Need to scale numerical features for ridge regression



# Combine numerical features with one-hot-encoded features

features = scipy.sparse.hstack((numerical_features, one_hot_features),format='csr') 

all_feature_names = np.hstack((numerical_feature_names,one_hot_names)) # Store names of all features for later interpretation



target_column = ['price_log1p'] # We will fit log(1 + price) 

target = data[target_column].values



# Perform train and test split of data

rand_seed = 51 # For other models we will use the same random seed, so that we're always using the same train-test split

features_train, features_test, target_train, target_test = train_test_split(

    features, target, test_size=0.2, random_state=rand_seed)
%%time

from sklearn import linear_model



ridge_fit = linear_model.RidgeCV(cv=5)

ridge_fit.fit(features_train, target_train)

print("RidgeCV found an optimal regularization parameter alpha =",ridge_fit.alpha_)

test_score_no_text = ridge_fit.score(features_test,target_test)

print("Test score for Ridge Regression without text features:", test_score_no_text)
from sklearn.feature_extraction.text import CountVectorizer



# Same train-test split as before (same random seed)

data_train, data_test = train_test_split(data, test_size=0.2, random_state=rand_seed)



training_corpus = data_train['name'].values # Only use the training set to define the features we are going to extract

vectorizer = CountVectorizer(min_df=3) 

# min_df is the minimum number of times a word needs to appear in the corpus in order to be assigned a vector

vectorizer.fit(training_corpus)

num_words = len(vectorizer.vocabulary_) # Total number of words 

print("Number of distinct words to be used as features:",num_words)
full_corpus = data['name'].values

word_features = vectorizer.transform(full_corpus) # This is a sparse matrix of our word-occurrence features 

words = vectorizer.get_feature_names() # The actual words corresponding to the columns of the above feature matrix

word_frequencies = np.array(word_features.sum(axis=0))[0] # The total number of occurrences of each word in the dataset

print("Shape of word-occurrence feature matrix:",word_features.shape)
num_non_text = features.shape[1]

features_with_text = scipy.sparse.hstack((features, word_features),format='csr') 

# We want to keep the feature matrix in a sparse format for efficiency

feature_names = np.hstack((all_feature_names, words))   



# Same train-test split as before (same random seed)

features_with_text_train, features_with_text_test, target_train, target_test = train_test_split(

    features_with_text, target, test_size=0.2, random_state=rand_seed)



num_features = num_non_text + num_words



print("Number of non-text features: ",num_non_text)

print("Number of vectorized text features (word occurrences): ",num_words)

print("Features shape including text features: ",features_with_text.shape)
%%time

from sklearn import linear_model



ridge_fit = linear_model.RidgeCV(cv=5)

ridge_fit.fit(features_with_text_train, target_train)

print("RidgeCV found an optimal regularization parameter alpha =",ridge_fit.alpha_)

test_score_with_text = ridge_fit.score(features_with_text_test,target_test)

print("Test score for Ridge Regression WITHOUT text features:", test_score_no_text)

print("Test score for Ridge Regression WITH text features:", test_score_with_text)
coefs = ridge_fit.coef_[0] # Coefficients of the linear fit



# I'll make a num_features-sized array of zeros, and then fill the indices corresponding to the word-occurrence features

# with the total number of counts for the word in that dataset. So features that don't correspond to words have 

# word_counts = 0.

num_features = features_with_text.shape[1]

word_counts = np.zeros(num_features, dtype=int)

word_counts[num_non_text:] = word_frequencies



# Make a DataFrame of feature names, coefficients, and word counts, and sort it by magnitude of the coefficient.

coef_df = pd.DataFrame(data={'names': feature_names, 'coefs': coefs, 'total_word_counts': word_counts})

coef_df_sorted = coef_df.reindex(coef_df['coefs'].abs().sort_values(ascending=False).index)
with pd.option_context('display.max_rows', None): 

    print(coef_df_sorted.head(200))
vectorizer = CountVectorizer(ngram_range=(1, 2), # Tokenize both one-word and two-word phrases

                             min_df=3, # Token should occur at least 3 times in training set

                             token_pattern="[a-zA-Z0-9]{1,30}" # Regular expression defining the form of a word or 1-gram

                            ) 

vectorizer.fit(training_corpus)

num_words = len(vectorizer.vocabulary_) # Total number of words 

print("Number of distinct tokens to be used as features:",num_words)
word_features = vectorizer.transform(full_corpus) # This is is a sparse matrix of our word-occurrence features 

words = vectorizer.get_feature_names() # The actual words corresponding to the columns of the above feature matrix

word_frequencies = np.array(word_features.sum(axis=0))[0] # The total number of occurrences of each word in the dataset



features_with_text = scipy.sparse.hstack((features, word_features),format='csr') 

feature_names = np.hstack((all_feature_names, words))   



features_with_text_train, features_with_text_test, target_train, target_test = train_test_split(

    features_with_text, target, test_size=0.2, random_state=rand_seed)



num_features = num_non_text + num_words



print("Number of non-text features: ",num_non_text)

print("Number of vectorized text features (word occurrences): ",num_words)

print("Features shape including text features: ",features_with_text.shape)
%%time

from sklearn import linear_model



ridge_fit = linear_model.RidgeCV(cv=5)

ridge_fit.fit(features_with_text_train, target_train)

print("RidgeCV found an optimal regularization parameter alpha =",ridge_fit.alpha_)

test_score_with_bigrams = ridge_fit.score(features_with_text_test,target_test)

print("Test score for Ridge Regression WITHOUT text features:", test_score_no_text)

print("Test score for Ridge Regression with single-word tokens:", test_score_with_text)

print("Test score for Ridge Regression with bigram tokens:", test_score_with_bigrams)
coefs = ridge_fit.coef_[0] # Coefficients of the linear fit



# I'll make a num_features-sized array of zeros, and then fill the indices corresponding to the word-occurrence features

# with the total number of counts for the word in that dataset. So features that don't correspond to words have 

# word_counts = 0.

num_features = features_with_text.shape[1]

word_counts = np.zeros(num_features, dtype=int)

word_counts[num_non_text:] = word_frequencies



# Make a DataFrame of feature names, coefficients, and word counts, and sort it by magnitude of the coefficient.

coef_df = pd.DataFrame(data={'names': feature_names, 'coefs': coefs, 'total_word_counts': word_counts})

coef_df_sorted = coef_df.reindex(coef_df['coefs'].abs().sort_values(ascending=False).index)



with pd.option_context('display.max_rows', None): 

    print(coef_df_sorted.head(200))