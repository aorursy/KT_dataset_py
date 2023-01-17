# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting
import seaborn as sns #plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Loading the data

train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')
# Exploration

print ('Exploring the training data')
print (train_df.head(5))
print (train_df.shape)
print (train_df.columns)
print (train_df.info())

print ('-----------------------------------------------------------------------------------------------')
print ('Exploring the test data')
print (test_df.head(5))
print (test_df.shape)
print (test_df.info())

print ('-----------------------------------------------------------------------------------------------')
print ('How many cuisine types are there? How common are they in the data set?')

unique_cuisine_types = train_df['cuisine'].nunique()
print ('There are %d unique cuisine types'  %  (unique_cuisine_types))

freq_cuisines = train_df['cuisine'].value_counts()
plt.figure(figsize=(20,6))
sns.barplot(x= freq_cuisines.index, y= freq_cuisines.values, color = 'b')
plt.xlabel('type of cuisine')
plt.ylabel('# of recipes')
plt.title('# of recipes per type of cuisine in training data')
# Exploration continued...

# 1. Counting the number of ingredients in each recipe list
number_of_ingredients = []
for i in range(len(train_df['ingredients'])):
    number_of_ingredients.append(len(train_df['ingredients'][i]))

train_df['number_of_ingredients'] = number_of_ingredients

print ('The average number of ingredients is %d' % np.average(number_of_ingredients))
print ('The max number of ingredients is %d' % np.max(number_of_ingredients))
print ('The min number of ingredients is %d' % np.min(number_of_ingredients))

# What do the ranges of # of ingredients look like for the different cuisines?

# Getting the min and max values of the boxplots to order it by size of range
lowIQ = train_df.groupby(['cuisine']).quantile(0.25)['number_of_ingredients']
highIQ = train_df.groupby(['cuisine']).quantile(0.75)['number_of_ingredients']
IQR = highIQ - lowIQ
minvalue = train_df.groupby(['cuisine']).min()['number_of_ingredients']
maxvalue = highIQ + (IQR * 1.5)
overall_range = maxvalue - minvalue
ordered_cuisines = (overall_range.sort_values(ascending = False).index)

plt.figure(figsize=(20,6))
sns.boxplot(x="cuisine", y="number_of_ingredients",data= train_df, width = 0.7, color = 'b', order = ordered_cuisines)
plt.xlabel('type of cuisine')
plt.ylabel('# of ingredients')
plt.title('spread of # of ingredients by type of cuisine, ordered by range in # of ingredients')
# Turn the ingredients into a single string so we can process them as individual words, as important words may be more easily recognised as common between 
# recipes.
train_df['seperated_ingredients'] = train_df['ingredients'].apply(','.join)
test_df['seperated_ingredients'] = test_df['ingredients'].apply(','.join)

#Splitting the training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df['seperated_ingredients'], train_df['cuisine'], test_size = 0.30, random_state = 102)
# Creating a pipeline

from sklearn.pipeline import Pipeline

#Vectorize imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
pattern = r"[A-Za-z]" 
from sklearn.preprocessing import MaxAbsScaler

# Set up a vectorizer which can be tested and words substituted into it.

vec = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words = 'english', lowercase = True, token_pattern = pattern, ngram_range = (1,2))
#vec = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words = 'english', lowercase = True, token_pattern = pattern, max_df = 0.1, ngram_range = (1,2))

#Classifier imports
#from sklearn.naive_bayes import MultinomialNB - 0.74
#from sklearn.neighbors import KNeighborsClassifier - 0.40
#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import validation_curve

C_param_range = [0.5,0.8,1,2,3]
accuracy_list = []
#Set up a classifier
for i in C_param_range:  
    classifier = LogisticRegression(penalty = 'l2', C = i,random_state = 0)
    # Create and fit the pipeline
    pl = Pipeline([
        ('vec',vec), 
        ('scale', MaxAbsScaler()),
        ('classifier', classifier)])
    pl.fit(X_train, y_train)
    accuracy = pl.score(X_test, y_test)
    print (accuracy)
    accuracy_list.append(accuracy)

zipped = list(zip(C_param_range,accuracy_list))
print (zipped)
#print ('The score is: %.5f '% accuracy)
# Best score so far is: 0.78856
# What is the maximum accuracy we have got so far and which parameter value does it correspond to?
from operator import itemgetter
from heapq import nlargest
import itertools

result = max(zipped,key=lambda x:x[1])
print (result[0])

classifier = LogisticRegression(penalty = 'l2', C = result[0],random_state = 0)

#Create and fit the pipeline
pl = Pipeline([
    ('vec',vec), 
    ('scale', MaxAbsScaler()),
    ('classifier', classifier)])
pl.fit(X_train, y_train)
y_predicted = pl.predict(X_test)
accuracy = pl.score(X_test, y_test)
print ('The score is: %.5f '% accuracy)
# Using a confusion matrix to identify areas of error - where could we focus a model fix?
from sklearn.metrics import confusion_matrix

cm = pd.DataFrame(confusion_matrix(y_test, y_predicted, labels=train_df['cuisine'].unique()), index=train_df['cuisine'].unique(), columns=train_df['cuisine'].unique())

totals = y_test.value_counts()
totals_from_training = y_train.value_counts()
joined = pd.concat([cm, totals], axis = 'columns', sort = False)
joined['totals_from_test'] = joined.iloc[:,-1]
joined['totals_from_training'] = totals_from_training
joined['%_correct'] = [round((joined.loc[i,i])/(joined.loc[i,'totals_from_test']),2) for i in joined.columns[:-3]]
joined
#Are the less common cuisines predicted correctly less of the time? 
from scipy.stats import pearsonr

correlation = pearsonr(joined['totals_from_training'], joined['%_correct'])
print ('The correlation score is: %.2f' % correlation[0])

order = (joined['totals_from_training'].sort_values(ascending = True).index)
#print (order)
plt.figure(figsize=(20,6))
sns.barplot(x= joined.index, y= joined["%_correct"], order = order)
plt.xlabel('type of cuisine')
plt.ylabel('%_correct')
plt.title('% of predictions correct by cuisine, ordered by cuisine frequency in test-set')
plt.show()
sns.regplot(x= joined['totals_from_training'], y= joined["%_correct"])
plt.xlabel('Total cuisine frequency in training set')
plt.ylabel('% correct in test set')
plt.show()

# Writing the test submission file 
submission = pl.predict(test_df['seperated_ingredients'])

submission_file = pd.DataFrame(data = submission, columns = ['cuisine'], index = test_df['id'])

submission_file.reset_index(level=0, inplace=True)

print (submission_file.head(5))
submission_file.to_csv('submission5.csv', index = False)