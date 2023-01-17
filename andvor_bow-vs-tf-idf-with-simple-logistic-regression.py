#First of all lets import training set and split it to DataFrame of independent variables and Series of dependent variable  



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', index_col=0)

lastindex = len(train_data.columns) -1

X = train_data.iloc[:,0:lastindex]

y = train_data.iloc[:,lastindex]
#import useful libraries

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



def preprocdata(df):

    #Fill empty categorical data

    #We will treat missing value as special categorical value

    df[['keyword','location']] = df[['keyword','location']].fillna('empty') 

    

    #Delete hyperlinks from text. After deleting clear data from symbols except english alphabet

    df['text'] = df['text'].map(lambda x: x.lower())

    df['text'] = df['text'].map(lambda x: re.sub(r"http[s]?:\/\/\S+\b","hypelinkplaceholder",x)) 

    df['text'] = df['text'].map(lambda x: re.sub(r"[^a-z ]","",x))

    

    #Delete stopwords and stem words

    stopwordsset = set(stopwords.words('english'))

    ps = PorterStemmer()

    df['text'] = df['text'].map(lambda x: ' '.join([ps.stem(word) for word in x.split(' ') if word not in stopwordsset])) 

    

    

    #Delete words encountered less than twice

    allwords = []

    text_as_list = list(df['text'].map(lambda x: x.split(' ')))

    [[allwords.append(word) for word in tweet] for tweet in text_as_list]    

    

    #Count each word frequency

    words_frequency = pd.Series(allwords,name="words").value_counts()

    FREQUENCY_THRESHOLD = 2 

    insignificantwords = list(words_frequency.loc[words_frequency<FREQUENCY_THRESHOLD].index)

    insignificantwords_set = set(insignificantwords)

    df['text'] = df['text'].map(lambda x: ' '.join([word for word in x.split(' ') if word not in insignificantwords_set]).strip())     

    

    

    # if after deleting empty unsignificant words text become empty replace it with empty placeholder

    df.loc[df.text=='',"text"] = "emptyplaceholder"
#Prepare our dataset and create clasifier

preprocdata(X)

classifier = LogisticRegression()
from sklearn.preprocessing import OneHotEncoder

keyword_encoder = OneHotEncoder(sparse=False, handle_unknown='error', drop='first')

encoded_categorical_values = keyword_encoder.fit_transform(X.iloc[:,0:1])

encoded_categorical_values_df = pd.DataFrame(encoded_categorical_values,columns = keyword_encoder.get_feature_names(), index = X.index)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

vectorisedtext =  vectorizer.fit_transform(X['text'])

#DataFrame with Bag of words

bow_df  = pd.DataFrame(vectorisedtext.todense(), columns = ["bow_" + featurename for featurename in vectorizer.get_feature_names()], index = X.index)

#DataFrame containing both keyword and Bag of words

X_bow = encoded_categorical_values_df.join(bow_df)



X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size = 0.2 )

score = cross_val_score(classifier,X_train, y_train, cv = 10, scoring = 'f1')

print('Bag of words f1 = {0:.3f} \u00b1 {1:.3f}'.format(score.mean(),score.std()))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

vectorisedtext = vectorizer.fit_transform(X['text'])

#DataFrame with TF-IDF

tfidf_df  = pd.DataFrame(vectorisedtext.todense(), columns = ["tfidf_" + featurename for featurename in vectorizer.get_feature_names()], index = X.index)

X_tfidf = encoded_categorical_values_df.join(tfidf_df)



X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size = 0.2 )

score = cross_val_score(classifier,X_train, y_train, cv = 10, scoring = 'f1')

print('TF-IDF f1 = {0:.3f} \u00b1 {1:.3f}'.format(score.mean(),score.std()))