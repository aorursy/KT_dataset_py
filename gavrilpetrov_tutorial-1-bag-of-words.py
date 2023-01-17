import pandas as pd

import os

import zipfile

import timeit

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
zf1 = zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip")

train = pd.read_csv(zf1.open("labeledTrainData.tsv"), sep="\t")

zf2 = zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip")

test = pd.read_csv(zf2.open("testData.tsv"), sep="\t")

train.drop(columns=['id'], inplace=True)

train.drop_duplicates(keep=False,inplace=True) 

#test.drop(columns=['id'], inplace=True)
import pandas as pd

imdb_master = pd.read_csv("../input/imdb-review-dataset/imdb_master.csv", encoding = "ISO-8859-1")

# Drop unneeded columns

imdb_master.drop(inplace=True, columns=['file','type','Unnamed: 0'])

# rename columns

imdb_master.columns = ['review','sentiment']

# keep only neg and positive reviews

imdb_master=imdb_master[(imdb_master.sentiment == "neg") | (imdb_master.sentiment == "pos")]

# rename base lables neg=0 and pos=1

di = {"neg": int("0"), "pos": int("1")}

imdb_master['sentiment']=imdb_master['sentiment'].map(di).astype('int64')

# remove duplicates

imdb_master.drop_duplicates(keep=False,inplace=True) 

# addup imdb_master and train

#train = pd.concat([imdb_master,train])

#train.reset_index(drop=True,inplace=True)
# Import BeautifulSoup into your workspace

from bs4 import BeautifulSoup             



# Initialize the BeautifulSoup object on a single movie review     

example1 = BeautifulSoup(train["review"][0])  



import re

# Use regular expressions to do a find-and-replace

letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for

                      " ",                   # The pattern to replace it with

                      example1.get_text() )  # The text to search



lower_case = letters_only.lower()        # Convert to lower case

words = lower_case.split()               # Split into words
import nltk

#nltk.download('stopwords')  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list

print(stopwords.words("english") )
# Remove stop words from "words"

words = [w for w in words if not w in stopwords.words("english")]
def review_to_words( raw_review ):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed movie review)

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review).get_text() 

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

num_reviews = train["review"].size



print ("Cleaning and parsing the training set movie reviews...\n")

# Initialize an empty list to hold the clean reviews

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in range( 0, num_reviews ):

    # If the index is evenly divisible by 1000, print a message

    if( (i+1)%1000 == 0 ):

        print("Review %d of %d\n" % ( i+1, num_reviews )    )                                                                



    # Call our function for each one, and add the result to the list of

    # clean reviews

    clean_train_reviews.append( review_to_words( train["review"][i] ) )
print("Creating the bag of words...\n")

from sklearn.feature_extraction.text import CountVectorizer



# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 5000) 



# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

train_data_features = vectorizer.fit_transform(clean_train_reviews)



# Numpy arrays are easy to work with, so convert the result to an 

# array

train_data_features = train_data_features.toarray()
#print("Training the random forest...")

from sklearn.ensemble import RandomForestClassifier



# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100,) 



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

%time forest = forest.fit( train_data_features, train["sentiment"] ) # 5 mins
model=forest
import xgboost as xgb

import lightgbm as lgb

xgboost = xgb.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 



lightgbm = lgb.LGBMClassifier()





#model = lightgbm
#%time model = model.fit( train_data_features, train["sentiment"], verbose=1 ) ## time 40s 
# Create an empty list and append the clean reviews one by one

num_reviews = len(test["review"])

clean_test_reviews = [] 



print("Cleaning and parsing the test set movie reviews...\n")

for i in range(0,num_reviews):

    if( (i+1) % 1000 == 0 ):

        print ("Review %d of %d\n" % (i+1, num_reviews))

    clean_review = review_to_words( test["review"][i] )

    clean_test_reviews.append( clean_review )



# Get a bag of words for the test set, and convert to a numpy array

test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()



# Use the random forest to make sentiment label predictions

result = model.predict_proba(test_data_features)



# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame( data={"id":test["id"], "sentiment":result[:,1]} )



# Use pandas to write the comma-separated output file

output.to_csv( "Bag_of_Words_model_random_forest.csv", index=False, quoting=3 )