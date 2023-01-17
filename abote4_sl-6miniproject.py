# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/tweets-sentiment-analysis/train.csv", encoding='ISO-8859-1')

train_data
#Lets have a look at some random tweets to gain more insights



rand_indexs = np.random.randint(1,len(train_data),50).tolist()

train_data["SentimentText"][rand_indexs]
# Handling Emoticons in tweets

# We are gonna find what emoticons are used in our dataset

import re

tweets_text = train_data.SentimentText.str.cat()

emos = set(re.findall(r" ([xX:;][-']?.) ",tweets_text))

emos_count = []

for emo in emos:

    emos_count.append((tweets_text.count(emo), emo))

sorted(emos_count,reverse=True)
HAPPY_EMO = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "

SAD_EMO = r" (:'?[/|\(]) "

print("Happy emoticons:", set(re.findall(HAPPY_EMO, tweets_text)))

print("Sad emoticons:", set(re.findall(SAD_EMO, tweets_text)))
import nltk

from nltk.tokenize import word_tokenize



# Uncomment this line if you haven't downloaded punkt before

# or just run it as it is and uncomment it if you got an error.

#nltk.download('punkt')

def most_used_words(text):

    tokens = word_tokenize(text)

    frequency_dist = nltk.FreqDist(tokens)

    print("There is %d different words" % len(set(tokens)))

    return sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)
most_used_words(train_data.SentimentText.str.cat())[:100]

#Stop Words
from nltk.corpus import stopwords



#nltk.download("stopwords")



mw = most_used_words(train_data.SentimentText.str.cat())

most_words = []

for w in mw:

    if len(most_words) == 1000:

        break

    if w in stopwords.words("english"):

        continue

    else:

        most_words.append(w)
sorted(most_words)
#Stemming
# I'm defining this function to use it in the 

# Data Preparation Phase

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer



#nltk.download('wordnet')

def stem_tokenize(text):

    stemmer = SnowballStemmer("english")

    stemmer = WordNetLemmatizer()

    return [stemmer.lemmatize(token) for token in word_tokenize(text)]



def lemmatize_tokenize(text):

    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(token) for token in word_tokenize(text)]


from sklearn.feature_extraction.text import TfidfVectorizer
#Building the pipeline



from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.pipeline import Pipeline
# We need to do some preprocessing of the tweets.

# We will delete useless strings (like @, # ...)

# because we think that they will not help

# in determining if the person is Happy/Sad



class TextPreProc(BaseEstimator,TransformerMixin):

    def __init__(self, use_mention=False):

        self.use_mention = use_mention

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        # We can choose between keeping the mentions

        # or deleting them

        if self.use_mention:

            X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")

        else:

            X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")

            

        # Keeping only the word after the #

        X = X.str.replace("#", "")

        X = X.str.replace(r"[-\.\n]", "")

        # Removing HTML garbage

        X = X.str.replace(r"&\w+;", "")

        # Removing links

        X = X.str.replace(r"https?://\S*", "")

        # replace repeated letters with only two occurences

        # heeeelllloooo => heelloo

        X = X.str.replace(r"(.)\1+", r"\1\1")

        # mark emoticons as happy or sad

        X = X.str.replace(HAPPY_EMO, " happyemoticons ")

        X = X.str.replace(SAD_EMO, " sademoticons ")

        X = X.str.lower()

        return X
# This is the pipeline that will transform our tweets to something eatable.

# You can see that we are using our previously defined stemmer, it will

# take care of the stemming process.

# For stop words, we let the inverse document frequency do the job

from sklearn.model_selection import train_test_split



sentiments = train_data['Sentiment']

tweets = train_data['SentimentText']



# I get those parameters from the 'Fine tune the model' part

vectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize, ngram_range=(1,2))

pipeline = Pipeline([

    ('text_pre_processing', TextPreProc(use_mention=True)),

    ('vectorizer', vectorizer),

])



# Let's split our data into learning set and testing set

# This process is done to test the efficency of our model at the end.

# You shouldn't look at the test data only after choosing the final model

learn_data, test_data, sentiments_learning, sentiments_test = train_test_split(tweets, sentiments, test_size=0.3)



# This will tranform our learning data from simple text to vector

# by going through the preprocessing tranformer.

learning_data = pipeline.fit_transform(learn_data)
#Model Selection
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import BernoulliNB, MultinomialNB



lr = LogisticRegression()

bnb = BernoulliNB()

mnb = MultinomialNB()



models = {

    'logitic regression': lr,

    'bernoulliNB': bnb,

    'multinomialNB': mnb,

}



for model in models.keys():

    scores = cross_val_score(models[model], learning_data, sentiments_learning, scoring="f1", cv=10)

    print("===", model, "===")

    print("scores = ", scores)

    print("mean = ", scores.mean())

    print("variance = ", scores.var())

    models[model].fit(learning_data, sentiments_learning)

    print("score on the learning data (accuracy) = ", accuracy_score(models[model].predict(learning_data), sentiments_learning))

    print("")
# Test
mnb.fit(learning_data, sentiments_learning)

lr.fit(learning_data, sentiments_learning)
bnb.fit(learning_data, sentiments_learning)
testing_data = pipeline.transform(test_data)

mnb.score(testing_data, sentiments_test)
bnb.score(testing_data, sentiments_test)
lr.score(testing_data, sentiments_test)
# Predicting on the test.csv

sub_data = pd.read_csv("../input/tweets-sentiment-analysis/test.csv", encoding='ISO-8859-1')

sub_learning = pipeline.transform(sub_data.SentimentText)

sub = pd.DataFrame(sub_data.ItemID, columns=("ItemID", "Sentiment"))

sub["Sentiment"] = mnb.predict(sub_learning)

print(sub)