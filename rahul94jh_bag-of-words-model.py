# load all necessary libraries

import pandas as pd

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer



pd.set_option('max_colwidth', 100)
documents = ["Gangs of Wasseypur is a great movie.", "Nawaz performance in Scared games is just amazing. ", "Ustad Zakir hussain is performing in new Delhi this evening on bollywood based theme." , 

             "The success of a movie depends on the performance of the actors.", "There are no new movies releasing this week.",

             "Manoj bajpayee is one of the finest movie actor of his genre.", "OTT is now the prefered medium for cinema lovers than tradition theatre.", 

             "Netflix is outperforming it's competions but Amazon prime not too far behind."]

print(documents)
def preprocess_text(text):

    'changes document to lower case and removes stopwords'



    # change sentence to lower case

    text = text.lower()



    # tokenize into words

    words = word_tokenize(text)



    # remove stop words

    words = [word for word in words if word not in stopwords.words("english")]



    # join words to make sentence

    text = " ".join(words)

    

    return text



documents = [preprocess_text(text) for text in documents]

print(documents)

vect = CountVectorizer()

model = vect.fit_transform(documents)

print(model) # returns the rown and column number of cells which have 1 as value
# print the full sparse matrix

print(model.toarray())
# get the shape of the matrix created, there are 45 unique words/features identified by the CountVectorizer

print(model.shape)

# get the feature names

print(vect.get_feature_names())
# read the file into a panda dataframe

filename = "../input/sms-spam-collection-dataset/spam.csv"

spam = pd.read_csv(filename,encoding='latin-1')

spam.head()
# drop unused columns

spam = spam.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

spam = spam.rename(columns={"v1":"label", "v2":"message"})

# Let's check the shape of DataFrame

print(spam.shape) #we have 5572 messages

spam.head()
# extract the messages from the dataframe

messages = spam.message

print(messages)
# convert messages into list

messages = [message for message in messages]

# preprocess messages using the preprocess function

messages = [preprocess_text(message) for message in messages]

print(messages)
# bag of words model

vect = CountVectorizer()

model = vect.fit_transform(messages)
# look at the dataframe

pd.DataFrame(model.toarray(), columns = vect.get_feature_names())
# Let's have a look on the features we got using the CountVectorizer

print(vect.get_feature_names())  # these features are the bag of words