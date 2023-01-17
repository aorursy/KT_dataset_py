import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



input_dir = '/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb'



train_dir = os.path.join(input_dir, 'train')

test_dir = os.path.join(input_dir, 'test')
# Check what is in these reviews

sample_review = ''

with open(os.path.join(train_dir, 'pos/121_10.txt')) as rev:

    

    lines = rev.readlines()

    

    sample_review = "\n".join(lines)



print(sample_review)
# strip off html tags

import re



# replace all tags with empty string

sample_review = re.sub(r"\<[^\>]*\>", '', sample_review)



print(sample_review)
# tokenize using tweet tokenizer to keep contractions

from nltk.tokenize import TweetTokenizer



tokenizer = TweetTokenizer()



sample_review_tokens = tokenizer.tokenize(sample_review)



print(sample_review_tokens)
# stop words removal

from nltk.corpus import stopwords



# get stopwords from english

sw = set(stopwords.words('english'))



# filter the review and remove stopwords

sample_review_tokens_filtered = []



for token in sample_review_tokens:

    if token not in sw:

        sample_review_tokens_filtered.append(token)



print(sample_review_tokens_filtered, end="\n\n")



print(len(sample_review_tokens), len(sample_review_tokens_filtered), sep="|")
# parts of speech tagging

import nltk



tagged_review_tokens = nltk.pos_tag(sample_review_tokens_filtered)



print(tagged_review_tokens)
# Lemmatization of the tokens using the pos tags

from nltk.corpus import wordnet

from nltk.stem.wordnet import WordNetLemmatizer



# function for converting treebank tags to wordnet tags

def get_wordnet_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN



# function to lemmatize tagged tokens

def lemmatize(token):

    word, tag = token

    

    lemmatizer = WordNetLemmatizer()

    

    if word not in list(".,;'\"-"):

        return lemmatizer.lemmatize(word, get_wordnet_pos(tag))

    

    return word



# apply to all items in the list

lemmatized_tokens = list(map(lemmatize, tagged_review_tokens))



print(lemmatized_tokens)
# get all positive and negetive review list



pos_reviews = os.listdir(os.path.join(train_dir, 'pos'))

neg_reviews = os.listdir(os.path.join(train_dir, 'neg'))



print('Positve:', len(pos_reviews))

print('Negetive:', len(neg_reviews))
# create a review list by combining these reviews

review_list = []



# get all positive reviews

for rev in pos_reviews:

    

    with open(os.path.join(train_dir, 'pos', rev)) as review:

        

        review_text = "\n".join(review.readlines())

        

        review_list.append(review_text)



print(len(review_list))        
# get all negetive reviews

for rev in neg_reviews:

    

    with open(os.path.join(train_dir, 'neg', rev)) as review:

        

        review_text = "\n".join(review.readlines())

        

        review_list.append(review_text)



print(len(review_list))
# strip off html tags

def strip_html_tags(text):

    

    return re.sub(r"\<[^\>]*\>", "", text)



review_list = list(map(strip_html_tags, review_list))



print(review_list[:3])
# tokenize the review list



def tokenize(text):

    tokenizer = TweetTokenizer()

    

    return tokenizer.tokenize(text)



# to avoid memory over use we will replace the same list each time

review_list = list(map(tokenize, review_list))



print(review_list[0])
# remove stop words from the token list



def stop_words_remove(tokens):

    

    # filter the review and remove stopwords

    tokens_filtered = []

    

    # sw is the set of stopwords in english

    for token in tokens:

        if token not in sw:

            tokens_filtered.append(token)

    

    return tokens_filtered



review_list = list(map(stop_words_remove, review_list))



print(review_list[0])
# pos tagging each token



def pos_tag_tokens(tokens):

    

    return nltk.pos_tag(tokens)



review_list = list(map(pos_tag_tokens, review_list))



print(review_list[0])
# lemmatize each token in each review

def lemmatize_review_tokens(tokens):

    

    # the lemmatize function is already defined

    return list(map(lemmatize, tokens))



review_list = list(map(lemmatize_review_tokens, review_list))



print(review_list[0])
# combine the tokens to form the sentence to create a list of documents to be vectorized

review_list = list(map(lambda tokens: " ".join(tokens), review_list))



print(*review_list[:3], sep="\n\n")
# remove extra spaces between the punctuations

def normalize_text(text):

    return re.sub(r"\s?([\.\,\/\!\-\"\'])\s?", r"\g<1>", text)



review_list = list(map(normalize_text, review_list))



print(*review_list[:3], sep="\n\n")
# vectorize



from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(lowercase=True, stop_words=list(sw), min_df=3)



X_count_vect = vectorizer.fit_transform(review_list)



print(X_count_vect.shape)
# convert vector to a pandas DataFrame



# get names of the features

X_names = vectorizer.get_feature_names()



# create the dataframe using the feature names

X_count_vect = pd.DataFrame(X_count_vect.toarray(), columns=X_names)



X_count_vect.head()
# create the targets



pos_targets = np.ones((12500,), dtype=np.int)

neg_targets = np.zeros((12500,), dtype=np.int)



# combine positive and negetive target list

target_list = []



target_list.extend(pos_targets)

target_list.extend(neg_targets)



# convert the targets into a pandas Series

y = pd.Series(target_list)



y.head()
# split into training and testing set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_count_vect, y, test_size=0.2, random_state=5)



print(len(X_train), len(X_test), sep="\n")
from sklearn.naive_bayes import MultinomialNB



# create the model

clf = MultinomialNB()



# fit the model

clf.fit(X_train, y_train)
# Lets test the accuracy of the model using the test data



from sklearn.metrics import accuracy_score



# predict using test data

y_pred = clf.predict(X_test)



# check the accuracy against actual value

print(accuracy_score(y_test, y_pred))