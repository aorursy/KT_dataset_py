import nltk

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import string

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import re

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn import linear_model

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from IPython.display import display, Image

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import requests

import random

import webbrowser

#nltk.download('wordnet')
# Reading in data, saving shape of train because we wil concatenate the

# test and train sets for some of the preprocessing

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train_shape = train.shape[0]

print(train.shape)

print(test.shape)
mbti = pd.concat([train.posts, test.posts])

print(mbti.shape)
train.info()

print('------------------------')

test.info()
print(train.isnull().any())

print(test.isnull().any())
my_colors ='g'

train.groupby('type').count().sort_values('posts',ascending=False).plot(kind='bar',stacked=True,colors=my_colors,figsize=(12,4))
mbti_type='INTP' # choose a type

stopwords = set(STOPWORDS) # set stopwords, so we can remove stopwords

words=train[train['type']==mbti_type]['posts'].sample(n=20) # randomly select 20 rows to use as example

wordcloud = WordCloud(background_color='white',stopwords=stopwords,max_words=200,max_font_size=40,scale=3,random_state=1 # chosen at random by flipping a coin; it was heads

         ).generate(str(words)) 

fig = plt.figure(1, figsize=(12, 12)) # set figure size

plt.imshow(wordcloud)

plt.show()
def keep_webs(arr):

    """ function takes in an array/ series of posts

        goes through the  posts, finds all the urls and saves them

        into  an array"""

    websites=[] #initialize empty array

    for i in arr: #loop through data

        urls =r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+' #regex for websites

        websites += re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',i)#appends into websites array the found url

    return websites
# mbti = pd.concat([train.posts, test.posts])

# urls=keep_webs(mbti)

# rand_num=random.randrange(0,len(urls)-1,1) # randomly generates an index for an url to be open

# webbrowser.open(list(set(keep_webs(mbti)))[rand_num])#opens randomly choosen url

# # the part is commented out because it opens the website
#removing web urls and converting all text to lowercase

def remove_url_lowr(ser):

    """

    Takes in a series of strings, and performs the following:

    

    1.Removes urls and replaces them with the text web-url, using regex

    2.Converts strings to all lower case

    

    ser: series of elements

    

    """

    

    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    subs_url = r'web-url'

    ser = ser.replace(to_replace = pattern_url, value = subs_url, regex = True)

    

    ser = ser.str.lower()

    

    return ser
def text_process(post):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Lemmatizes the the bag of words

    4. Returns a list of the cleaned, processed text

    """



    # Check characters to see if they are in punctuation

    punc_free = [char for char in post if char not in string.punctuation]



    # Join the characters again to form the string.

    punc_free = ''.join(punc_free)

    

    # Remove any stopwords

    bag = [word for word in punc_free.split() if word.lower() not in stopwords]

    

    #lemmatize the bag of words

    lemmatizer = WordNetLemmatizer()

    

    return [lemmatizer.lemmatize(word) for word in bag]
mbti = remove_url_lowr(mbti)
train.type.value_counts()
def resampling_balance(class_type, n, bar=True):

    """

    

    Takes in a dataframe filtered by the majority or minority class 

    and uses the resample function to resample to n samples.

    

    class_type: filtered dataframe by majority or minority class

    n: number of samples

    bar: boolean value, default=True, True when upsampling, False when downsampling

    

    Output:

    """

    

    type_resampled = resample(class_min, 

                                 replace=bar,  

                                 n_samples=n,  

                                 random_state=42)

    

    return type_resampled
# binarizing our labels- a function to look for a letter at index n, if found apply 1, if not apply 0

train['mind'] = train['type'].apply(lambda x: x[0] == 'E').astype('int')

train['energy'] = train['type'].apply(lambda x: x[1] == 'N').astype('int')

train['nature'] = train['type'].apply(lambda x: x[2] == 'T').astype('int')

train['tactics'] = train['type'].apply(lambda x: x[3] == 'J').astype('int')
# assigning the new labels to target value names

y_m = train['mind']

y_e = train['energy']

y_n = train['nature']

y_t = train['tactics']
#Splitting our datasets again

X_train = mbti[:train_shape]

X_test = mbti[train_shape:]



print(X_train.shape)

print(X_test.shape)
#TFIDF vectorizer with text process as the analyzer

#count vectorizer

cvc = CountVectorizer(analyzer=text_process)

bag1 = cvc.fit_transform(X_train)

bag2 = cvc.transform(X_test)



# Print total number of vocab words

print(len(cvc.vocabulary_))
#TF-IDF

tfidf = TfidfTransformer()

X_train = tfidf.fit_transform(bag1)

X_test = tfidf.transform(bag2)



print(X_train.shape)

print(X_test.shape)
# for logistic regression - this function runs a gridsearch on the paramaters 

# that we've specified for this model. It will also run through the grid of weights to 

# determine optimal class weights



def gridsearch_tuning(estimator, X, y, params={}):

    gsc = GridSearchCV(estimator=estimator,param_grid=params,scoring='f1',cv=3,n_jobs=-1)

    

    grid_result = gsc.fit(X, y)



#     print("Best parameters : %s" % grid_result.best_params_)



#     # Plot the weights vs f1 score

#     dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],

#                        'weight': weights })

#     dataz.plot(x='weight')

    

    return grid_result.best_params_
#paramaters for logistic regression

weights = np.linspace(0.05, 0.95, 20)

params_log={

    'class_weight': [{0: x, 1: 1.0-x} for x in weights],

    'C': [0.001, 0.01, 0.1, 1, 10, 100]

}



#parameters for KNN

k_range = list(range(1,31))

weight_options = ["uniform", "distance"]

param_grid_knn = dict(n_neighbors = k_range, weights = weight_options)



#parameters for MultinomialNB

param_nb={

    'alpha': np.linspace(0.5, 1.5, 6),

    'fit_prior': [True, False]

}
# running the gridsearch function on all 4 labels

mind_tune = gridsearch_tuning(linear_model.LogisticRegression(), X_train, y_m, params_log)

energy_tune = gridsearch_tuning(linear_model.LogisticRegression(), X_train, y_e, params_log)

nature_tune = gridsearch_tuning(linear_model.LogisticRegression(), X_train, y_n, params_log)

tactics_tune = gridsearch_tuning(linear_model.LogisticRegression(), X_train, y_t, params_log)





# mind_tune_knn = gridsearch_tuning(KNeighborsClassifier(), X_train, y_m)

# energy_tune_knn = gridsearch_tuning(KNeighborsClassifier(), X_train, y_e)

# nature_tune_knn = gridsearch_tuning(KNeighborsClassifier(), X_train, y_n)

# tactics_tune_knn = gridsearch_tuning(KNeighborsClassifier(), X_train, y_t)



# mind_tune_dct = gridsearch_tuning(DecisionTreeClassifier(), X_train, y_m)

# energy_tune_dct = gridsearch_tuning(DecisionTreeClassifier(), X_train, y_e)

# nature_tune_dct = gridsearch_tuning(DecisionTreeClassifier(), X_train, y_n)

# tactics_tune_dct = gridsearch_tuning(DecisionTreeClassifier(), X_train, y_t)
# Creating our model object

# Adding the dict of best params returned from gridsearch as **kwargs

lr_m = linear_model.LogisticRegression(**mind_tune)

lr_e = linear_model.LogisticRegression(**energy_tune)

lr_n = linear_model.LogisticRegression(**nature_tune)

lr_t = linear_model.LogisticRegression(**tactics_tune)



# knn_m = KNeighborsClassifier(**mind_tune_knn)

# knn_e = KNeighborsClassifier(**energy_tune_knn)

# knn_n = KNeighborsClassifier(**nature_tune_knn)

# knn_t = KNeighborsClassifier(**tactics_tune_knn)



# dct_m = DecisionTreeClassifier(**mind_tune_dct)

# dct_e = DecisionTreeClassifier(**energy_tune_dct)

# dct_n = DecisionTreeClassifier(**nature_tune_dct)

# dct_t = DecisionTreeClassifier(**tactics_tune_dct)
#our final models



# Fitting

lr_m.fit(X_train, y_m)

lr_e.fit(X_train, y_e)

lr_n.fit(X_train, y_n)

lr_t.fit(X_train, y_t)



# Predicting

y_predm = lr_m.predict(X_test)

y_prede = lr_e.predict(X_test)

y_predn = lr_n.predict(X_test)

y_predt = lr_t.predict(X_test)
# Creating df from the predicted values and outputting to csv

final_mbti = pd.DataFrame(dict(Id=test.id, mind=y_predm, energy=y_prede, nature=y_predn, tactics=y_predt))

final_mbti.rename(columns={'Id': 'id'}, inplace=True)



final_mbti.to_csv('mbti_logistic.csv', index=False)