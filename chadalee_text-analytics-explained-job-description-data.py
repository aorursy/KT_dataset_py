# import libraries
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize

import re

import matplotlib.pyplot as plt
%matplotlib inline

import math

import warnings
warnings.filterwarnings('ignore')
# Read in the train_rev1 datafile downloaded from kaggle
df = pd.read_csv('../input/Train_rev1.csv')
df.head(2)
print("Data Shape:", df.shape)
# randomly sample 10000 rows from the data
import random
random.seed(1)
indices = df.index.values.tolist()

random_10000 = random.sample(indices, 10000)

random_10000[:5]
# subset the imported data on the selected 2500 indices
train = df.loc[random_10000, :]
train = train.reset_index(drop = True)
train.head(2)
train.columns.values
# some problems with the way FullDescription has been encoded
def convert_utf8(s):
    return str(s)

train['FullDescription'] = train['FullDescription'].map(convert_utf8)
train.loc[2, 'FullDescription']
# Remove the urls first - Anything that has .com, .co.uk or www. is a url!
def remove_urls(s):
    s = re.sub('[^\s]*.com[^\s]*', "", s)
    s = re.sub('[^\s]*www.[^\s]*', "", s)
    s = re.sub('[^\s]*.co.uk[^\s]*', "", s)
    return s

train['Clean_Full_Descriptions'] = train['FullDescription'].map(remove_urls)
# Remove the star_words
def remove_star_words(s):
    return re.sub('[^\s]*[\*]+[^\s]*', "", s)

train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(remove_star_words)
def remove_nums(s):
    return re.sub('[^\s]*[0-9]+[^\s]*', "", s)

train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(remove_nums)
# Remove the punctuations
from string import punctuation

def remove_punctuation(s):
    global punctuation
    for p in punctuation:
        s = s.replace(p, '')
    return s

train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(remove_punctuation)
# Convert to lower case
train['Clean_Full_Descriptions'] = train['Clean_Full_Descriptions'].map(lambda x: x.lower())
print(nltk.pos_tag(word_tokenize("I love riding my motorcycle")))
# define a function for parts of speech tagging
# make a corpus of all the words in the job description
corpus = " ".join(train['Clean_Full_Descriptions'].tolist())

# This is the NLTK function that breaks a string down to its tokens
tokens = word_tokenize(corpus)

# Get the parts of speech tag for all words
answer = nltk.pos_tag(tokens)
answer_pos = [a[1] for a in answer]

# print a value count for the parts of speech
all_pos = pd.Series(answer_pos)
all_pos.value_counts().head()
# store english stopwords in a list
from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')

# define a function to remove stopwords from descriptions
def remove_stopwords(s):
    global en_stopwords
    s = word_tokenize(s)
    s = " ".join([w for w in s if w not in en_stopwords])
    return s

# Create a new column of descriptions with no stopwords
train['Clean_Full_Descriptions_no_stop'] = train['Clean_Full_Descriptions'].map(remove_stopwords)

# make a corpus of all the words in the job description
corpus = " ".join(train['Clean_Full_Descriptions_no_stop'].tolist())

# This is the NLTK function that breaks a string down to its tokens
tokens = word_tokenize(corpus)

answer = nltk.pos_tag(tokens)
answer_pos = [a[1] for a in answer]

all_pos = pd.Series(answer_pos)
all_pos.value_counts().head()
# prepare corpus from the descriptions that still have stopwords
corpus = " ".join(train['Clean_Full_Descriptions'].tolist())

#tokenize words
tokenized_corpus = nltk.word_tokenize(corpus)
fd = nltk.FreqDist(tokenized_corpus)

# get the top words
top_words = []
for key, value in fd.items():
    top_words.append((key, value))

# sort the list by the top frequencies
top_words = sorted(top_words, key = lambda x:x[1], reverse = True)

# keep top 100 words only
top_words = top_words[:100]

# Keep the frequencies only from the top word series
top_word_series = pd.Series([w for (v,w) in top_words])
top_word_series[:5]

# get actual ranks of these words - wherever we see same frequencies, we give same rank
word_ranks = top_word_series.rank(method = 'min', ascending = False)
# Get the value of the denominator n*x_n
denominator = max(word_ranks)*min(top_word_series)

# Y variable is the log of word ranks and X is the word frequency divided by the denominator
# above
Y = np.array(np.log(word_ranks))
X = np.array(np.log(top_word_series/denominator))

# fit a linear regression to these, we dont need the intercept!
from sklearn import linear_model
reg_model = linear_model.LinearRegression(fit_intercept = False)
reg_model.fit(Y.reshape(-1,1), X)
print("The value of theta obtained is:",reg_model.coef_)

# make a plot of actual rank obtained vs theoretical rank expected
plt.figure(figsize = (8,5))
plt.scatter(Y, X, label = "Actual Rank vs Frequency")
plt.title('Log(Rank) vs Log(Frequency/nx(n))')
plt.xlabel('Log Rank')
plt.ylabel('Log(Frequency/nx(n))')

plt.plot(reg_model.predict(X.reshape(-1,1)), X, color = 'red', label = "Zipf's law")
plt.legend()
# import the necessary functions from the nltk library
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

# prepare corpus from the descriptions that dont have stopwords
corpus = " ".join(train['Clean_Full_Descriptions_no_stop'].tolist())

#tokenize words
tokenized_corpus = nltk.word_tokenize(corpus)

# lemmatize these tokens
lemmatized_tokens = [lmtzr.lemmatize(token) for token in tokenized_corpus]

# word frequencies for the lemmatized tokens
fd = nltk.FreqDist(lemmatized_tokens)

# get the top words
top_words = []
for key, value in fd.items():
    top_words.append((key, value))

# sort the list by the top frequencies
top_words = sorted(top_words, key = lambda x:x[1], reverse = True)

# keep top 10 words only
top_words = top_words[:10]

top_words
# get the 75th percentile value of salary!
sal_perc_75 = np.percentile(train['SalaryNormalized'], 75)

# make a new target variable that captures whether salary is high (1) or low (0)
train['Salary_Target'] = np.where(train['SalaryNormalized'] >= sal_perc_75, 1, 0)
train.dtypes.value_counts()
train.isnull().sum()[train.isnull().sum()>0]
# this gives us the categorical variables and the number of unique entries in them!
train.select_dtypes('object').nunique(dropna = False)
exp_cities = ['London', 'Oxford', 'Brighton', 'Cambridge', 'Bristol', 'Portsmouth', 
              'Reading', 'Edinburgh', 'Leicester', 'York', 'Exeter']
def check_city(s):
    '''Given a Normalized Location this tells us if it is an expensive city'''
    global exp_cities
    answer = pd.Series(exp_cities).map(lambda x: x in s)
    answer = min(np.sum(answer),1)
    return answer

# add the indicator as a column in the dataframe
train['Exp_Location'] = train['LocationNormalized'].map(check_city)
train['Salary_Target'].value_counts()/len(train)
# Subset the columns required
columns_required = ['ContractType', 'ContractTime', 'Company', 'Category', 'SourceName', 'Exp_Location', 'Salary_Target']
train_b1 = train.loc[:, columns_required]

# Convert the categorical variables to dummy variables
train_b1 = pd.get_dummies(train_b1)

# Lets separate the predictors from the target variable
columns_selected = train_b1.columns.values.tolist()
target_variable = ['Salary_Target']

# predictors are all variables except for the target variable
predictors = list(set(columns_selected) - set(target_variable))

# setup the model
from sklearn.naive_bayes import BernoulliNB

X = np.array(train_b1.loc[:,predictors])
y = np.array(train_b1.loc[:,target_variable[0]])

# create test train splits 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

model = BernoulliNB()

# Fit the model and predict the output on the test data
model.fit(X_train, y_train)

# Predicted output
predicted = model.predict(X_test)

# Accuracy
from sklearn import metrics

print("Model Accuracy is:", metrics.accuracy_score(y_test, predicted))
print("Area under the ROC curve:", metrics.roc_auc_score(y_test, predicted))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, predicted))
# Lets lemmatize the job descriptions before we run the model
def text_lemmatizer(s):
    '''Given a description, this lemmatizes it'''
    tokenized_corpus = nltk.word_tokenize(s)
    
    # lemmatize
    s = " ".join([lmtzr.lemmatize(token) for token in tokenized_corpus])
    return s

# lemmatize the descriptions
train['Clean_Full_Descriptions_no_stop_lemm'] = train['Clean_Full_Descriptions_no_stop'].map(text_lemmatizer)

# make the X and y matrices for model fitting
X = np.array(train.loc[:, 'Clean_Full_Descriptions_no_stop_lemm'])
y = np.array(train.loc[:, 'Salary_Target'])

# split into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

# Convert the arrays into a presence/absence matrix
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
nb_mult_model = MultinomialNB().fit(X_train_counts, y_train)
predicted = nb_mult_model.predict(X_test_counts)

print("Model Accuracy:", metrics.accuracy_score(y_test, predicted))
print("Area under the ROC curve:", metrics.roc_auc_score(y_test, predicted))
print("Model Confusion Matrix:\n", metrics.confusion_matrix(y_test, predicted))
# Calculate the frequencies of words using the TfidfTransformer
X_train_bern = np.where(X_train_counts.todense() > 0 , 1, 0)
X_test_bern = np.where(X_test_counts.todense() > 0, 1, 0)

# Fit the model
from sklearn.naive_bayes import BernoulliNB
nb_bern_model = BernoulliNB().fit(X_train_bern, y_train)
predicted = nb_bern_model.predict(X_test_bern)

# print the accuracies
print("Model Accuracy:", metrics.accuracy_score(y_test, predicted))
print("Area under the ROC curve:", metrics.roc_auc_score(y_test, predicted))
print("Model Confusion Matrix:\n", metrics.confusion_matrix(y_test, predicted))
# extract the column names for the columns in our training dataset.
column_names = [x for (x,y) in sorted(count_vectorizer.vocabulary_.items(), key = lambda x:x[1])]

# probability of high salary
p_1 = np.mean(y_train)

# probability of low salary
p_0 = 1 - p_1

# create an array of feature vectors
feature_vectors = np.array(X_train_bern)

# probability of word appearance
word_probabilities = np.mean(feature_vectors, axis = 0)

# probability of seeing these words for class= 1 and class = 0 respectively
p_x_1 = np.mean(feature_vectors[y_train==1, :], axis = 0)
p_x_0 = np.mean(feature_vectors[y_train==0, :], axis = 0)

# words that are good indicators of high salary (class = 1)
high_indicators = p_x_1 * (np.log2(p_x_1) - np.log2(word_probabilities) - np.log2(p_1))

high_indicators_series = pd.Series(high_indicators, index = column_names)

# words that are good indicators of low salary (class = 0)
low_indicators = p_x_0 * (np.log2(p_x_0) - np.log2(word_probabilities) - np.log2(p_0))

low_indicators_series = pd.Series(low_indicators, index = column_names)
low_indicators_series[[i for i in low_indicators_series.index if i not in en_stopwords]].sort_values(ascending = False)[:10].index
high_indicators_series[[i for i in high_indicators_series.index if i not in en_stopwords]].sort_values(ascending = False)[:10].index