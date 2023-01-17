import sys, os, re, csv, codecs, numpy as np, pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from bs4 import BeautifulSoup 

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords # Import the stop word list

import gc

from sklearn.model_selection import train_test_split

from sklearn.externals import joblib
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

jokes = pd.read_csv('../input/jokes.csv')

submit_template = pd.read_csv('../input/sample_submission.csv', header = 0)
train.shape
train.head()
jokes['joke_text'][1]
def review_to_words( raw_review ):

    # Function to convert a raw sentence to a string of words

    # The input is a single string (a raw sentence), and 

    # the output is a single string (a preprocessed sentence)

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review,).get_text() 

    #

    # 2. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    #

    # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                                             

    #

    # 4. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # 

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))   

# Get the number of reviews based on the dataframe column size

num_reviews = jokes["joke_text"].size



# Initialize an empty list to hold the clean reviews

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

print ("Cleaning and parsing the training set of jokes...\n")

clean_train_reviews = []

for i in range( 0, num_reviews ):

    # If the index is evenly divisible by 10000, print a message

    if( (i+1)%20 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_reviews ))                                                                 

    jokes["joke_text"][i]=review_to_words( jokes["joke_text"][i])

print("Done")
jokes.head()
train_data = train.merge(jokes, on='joke_id', how='left')

train_data.head()
test_data = test.merge(jokes, on='joke_id', how='left')

test_data.head()
print ("Creating the bag of words...\n")

from sklearn.feature_extraction.text import CountVectorizer



# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 500) 



# The input to fit_transform should be a list of strings.

train_data_features = vectorizer.fit_transform(train_data['joke_text'])

##file='vectorizer.pkl'

#joblib.dump(vectorizer,'vectorizer.pkl')

# Numpy arrays are easy to work with, so convert the result to an 

# array

train_data_features = train_data_features.toarray()

##Zfile='train_data_features.pkl'

#joblib.dump(train_data_features,'train_data_features.pkl')



print ("Done")
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(jokes['joke_text'])
#splitting dataset into training and testing data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train_data_features,train_data["Rating"],test_size=0.2,random_state=0)

print("Splitting Done")
import lightgbm as lgb



lgb_train = lgb.Dataset(x_train, y_train)

lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)



# specify your configurations as a dict

params = {'num_leaves': 45,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.015,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4950}



print('Starting training...')

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=50,

                valid_sets=lgb_eval,

                early_stopping_rounds=5)

#file='gbm.pkl'

#joblib.dump(gbm,'gbm.pkl')
test_data_features = vectorizer.transform(test_data['joke_text'])



# Numpy arrays are easy to work with, so convert the result to an array

test_data_features = test_data_features.toarray()

print ("Done")
y_submit_new = gbm.predict(test_data_features)

print("Predictions Done")
y_submit_new[np.isnan(y_submit_new)]=0

sample_submission = submit_template

sample_submission[["Rating"]] = y_submit_new

sample_submission.to_csv('submission.csv', index=False)