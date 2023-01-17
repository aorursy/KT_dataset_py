# Load modules

import pandas as pd # Data manipulation

import matplotlib.pyplot as plt # Data visualization

import seaborn as sns

from bs4 import BeautifulSoup

import re

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

%matplotlib inline
# Read in the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')



# Function to convert tweet text to bag of words :)

def tweet_to_words(raw_tweet):

    tweet_text = BeautifulSoup(raw_tweet, "lxml").get_text() # Remove HTML

    letters_only = re.sub("[^a-zA-Z]", " ", tweet_text) # Remove any non-letter characters

    words = letters_only.lower().split() # Convert to lower case and split around spaces

    stops = set(stopwords.words("english")) # Make set from list -- faster to search!

    meaningful_words = [w for w in words if not w in stops] # Remove stopwords

    return(" ".join(meaningful_words)) # Join into one string around spaces



# List all identities

identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']



# Make list of words from tweets

def list_of_words(data):



    # Ensure that comment_text entries are all strings

    data['comment_text'] = data['comment_text'].astype(str)

    

    num_tweets = data["comment_text"].size

    clean_tweets = []



    for i in range(0, num_tweets):

        clean_tweets.append(tweet_to_words(data["comment_text"][i]))

        if(i % 500 == 0):

            print("at observation {}".format(i))

            

    return clean_tweets



# Convert list of tweets to features based on bag of words of size n

def word_counts(clean_tweets, n):

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=n)

    train_data_features = vectorizer.fit_transform(clean_tweets)

    return train_data_features.toarray()
# Smaller

#smaller_train = train.head(5000) # Just use 50,000 rows for now

#smaller_clean = list_of_words(smaller_train)

#clean_counts = word_counts(smaller_clean, 200)



# All training data

clean = list_of_words(train)

clean_counts = word_counts(smaller_clean, 200)
# Do this to fix weird error

lab_enc = preprocessing.LabelEncoder()

training_scores_encoded = lab_enc.fit_transform(train["target"]) # Use smaller_train if doing so



print("scores encoded")



forest = RandomForestClassifier(n_estimators = 50) # 100 trees

print("forest made")

forest = forest.fit(clean_counts, training_scores_encoded) # Features are word vectors, target is toxicity

print("forest fitted")
test_clean = list_of_words(test)

test_features = word_counts(test_clean, 200)



# Use random forest to make sentiment label predictions

result = forest.predict(test_features)



# Copy the results to a pandas dataframe with an "id" column and a "sentiment" column

output = pd.DataFrame(data = {"id":test["id"], "prediction":result})



# Use pandas to write output csv file

output.to_csv("submission.csv", index=False, quoting=3)