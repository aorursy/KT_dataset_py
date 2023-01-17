#get the data

import numpy as np

import pandas as pd



#text-processing

import string

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



#preparing data for the model

from sklearn.model_selection import train_test_split



#creating the model

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
def read_data(x):

    yelp = pd.read_csv('../input/'+x+'.csv')

    yelp = yelp[(yelp['stars']==1) | (yelp['stars']==5)][['stars','text']]

    yelp['length'] = yelp['text'].apply(len)

    return yelp
yelp = read_data('yelp')
yelp.head()
yelp['text'].head()
def text_pre_processing(x):

    nopunc = x.translate(str.maketrans('','',string.punctuation))

    nostop = [i for i in nopunc.split() if i not in stopwords.words('english')]

    nostop = ' '.join(nostop)

    porter = PorterStemmer()

    stemmed = [porter.stem(i) for i in nostop.split(' ')]

    cleaned_sentence = ' '.join(stemmed)

    return cleaned_sentence
yelp['text'] = yelp['text'].apply(text_pre_processing)
yelp.head()
def data_preparation(x):

    

    X = x['text']

    y = x['stars']

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    

    train_stars_1 = len(y_train[y_train==1])

    train_stars_5 = len(y_train[y_train==5])

    test_stars_1 = len(y_test[y_test==1])

    test_stars_5 = len(y_test[y_test==5])

    total_train = train_stars_1 + train_stars_5

    total_test = test_stars_1 + test_stars_5

    total_text = len(X_train)+len(X_test)



    print(f'length of text train: {len(X_train)}/{total_text}\nlength of text test: {len(X_test)}/{total_text}\n')

    print(f'Training stars:\nno. 1 star reviews: {train_stars_1}/{total_train}\nno. 5 star reviews: {train_stars_5}/{total_train}\n')

    print(f'Test stars:\nno. 1 star reviews: {test_stars_1}/{total_test}\nno. 5 star reviews: {test_stars_5}/{total_test}')

    

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = data_preparation(yelp)
def model_train_fit_predict(X_train, X_test, y_train, y_test):

    

    #instantiate the model

    pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_pre_processing)),

                     ('classifier', MultinomialNB())])

    

    #fit the model to the training data

    pipeline.fit(X_train, y_train)

   

    #use the model to predict the stars rating

    predictions = pipeline.predict(X_test)

   

    #create a dataframe to compare the predicted results with the actual rating

    error = pd.DataFrame()

    err = y_test-predictions

    error['error'] = err

    

    error['error'] = error['error'].replace(to_replace=0, value='correct')

    error['error'] = error['error'].replace(to_replace=-4, value='over-estimate')

    error['error'] = error['error'].replace(to_replace=4, value='under-estimate')

    

    print(f'Results:\n{error["error"].value_counts()}')

    

    #use a classification report to evaluate the model's performance

    report = classification_report(y_test,predictions)

    

    return report

    
results_report = model_train_fit_predict(X_train,X_test,y_train,y_test)
results_report