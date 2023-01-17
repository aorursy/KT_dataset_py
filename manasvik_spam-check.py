# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords



from sklearn.linear_model import LinearRegression,Ridge,Lasso,BayesianRidge

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/fake.csv')

print(data.describe())

print(data.head())
#columns with missing values

column_names = list(data.columns)

print(column_names)

for column in column_names:

    if data[column].count()<len(data):

        print(column,data[column].count())

        



#filling in missing values

#since title==thread_title for all existing (non-null) values of title I will simply be using thread_title 





#In case of text I will replace the missing values with '' to avoid any errors during tfidf calculation

data.loc[data['text'].isnull(),'text'] = ''





#country.. replacing the missing values with the most frequent country i.e. US

data.loc[data['country'].isnull(),'country'] = 'US' 







#In case of author the simplest (although not the best) solution would be to replace the missing values

#with the most frequent author.. in this case 'admin'



#I have decided to replace the missing values based on the country of the author.. maybe not a very good 

#idea but worth a shot :P



data.loc[data['author'].isnull(),'author'] = data.loc[data['author'].isnull(),'country']







#thread title.. similar to text.. replacing missing values with ''



data.loc[data['thread_title'].isnull(),'thread_title'] = ''





#domain_rank Here I will go for median substitution However since nearly a third of the data is missing 

#in this column this might not be very fruitful



data.loc[data['domain_rank'].isnull(),'domain_rank'] = np.nanmedian(np.array(data['domain_rank']))
# starting with features from tweet content

pstem = PorterStemmer()

stop = set(stopwords.words('english'))



def stemmer(text):

    wordlist = text.strip().split()

    pstem = PorterStemmer()

    j = '';

    for word in wordlist:

        try:

            j = j+pstem.stem(word)+' '

        except: 

            print(text)

    return j

#converting all texts and thread_titles to lowercase

data['text_lower'] = data['text'].map(lambda x: re.sub(r'[^a-z ]','',x.lower()))

data['text_lower'] = data['text_lower'].map(lambda x: ' '.join([word for word in x.strip().split() if word not in stop ]))

data['text_lower'] = data['text_lower'].map(lambda x: stemmer(x))

#data['text_lower'] = data['text_lower'].map(lambda x: ' '.join([pstem.stem(word) for word in x]))



data['thread_lower'] = data['thread_title'].map(lambda x: re.sub(r'[^a-z ]','',x.lower()))

data['thread_lower'] = data['thread_lower'].map(lambda x: ' '.join([word for word in x.strip().split() if word not in stop ]))

data['thread_lower'] = data['thread_lower'].map(lambda x: ' '.join([pstem.stem(word) for word in x.strip().split()]))



#feature1- number of words

data['num_words'] = data['text_lower'].map(lambda x: len(str(x).strip().split()))

 
#feature2 - similarity score between tweet text and title 

# I will be using sklearn for this



def similarity(t1,t2):

    t = [t1,t2]

    tfidf_vectorizer = TfidfVectorizer()

    try:

        vectors = tfidf_vectorizer.fit_transform(t)

        vectors = vectors.toarray()

        sim = cosine_similarity(vectors[0].reshape(1,-1),vectors[1].reshape(1,-1))[0]

    except:

        sim = 0 #error when empty vocabulary hence taking similarity as zero

    return sim



data['similarity'] = [similarity(b['text_lower'],b['thread_lower']) for (a,b) in data.iterrows()]

data['similarity'].describe() 
#mapping country to numbers

countries = list(data['country'].unique())

data['country_number'] = data['country'].map(lambda x: countries.index(x))

data['country_number'].describe()
#mapping languages to numbers

languages = list(data['language'].unique())

data['language_number'] = data['language'].map(lambda x: languages.index(x))

data['language_number'].describe()
feature_list = ['num_words','similarity','country_number','likes','shares','comments','replies_count','participants_count','domain_rank','language_number']

target1 = 'spam_score'

target2 = 'type'