import gzip

import nltk

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# from nltk.tokenize import word_tokenize, sent_tokenize

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LinearRegression
# Extract and read the input file line by line

def read_input_file(path):

    g = gzip.open(path, 'rb')

    for line in g:

        yield eval(line)
# Store the data in input file in a dataframe

def parse_input(path):

    i = 0

    df = {}

    for d in read_input_file(path):

        df[i] = d

        i += 1

    return pd.DataFrame.from_dict(df, orient='index')
# Preprocessing the input

def preprocess_input(data):



    # Merge summary and reviewText

    data['review'] = data['summary'] + ' ' + data['reviewText']



    # Separate out upvotes and downvotes from 'helpful'

    votes = list(zip(*list(data['helpful'].values)))

    data['upvotes'] = np.array(votes[0])

    data['downvotes'] = np.array(votes[1])

    

    # Remove unnecessary features

    del data['reviewTime'], data['unixReviewTime'], data['reviewerName'], data['summary'], data['reviewText'], data['helpful']



    # Rearrange columns

    data = data[['reviewerID', 'asin', 'overall', 'upvotes', 'downvotes', 'review']]

    

    return data
# Filtering the dataset

def filter_input(data):

    

    # (1) Review should have more than 10 votes

    data.drop(data[data.upvotes + data.downvotes <= 10].index, inplace=True)



    # (2) Each product should have more than 15 reviews

    product_review_count = data['asin'].value_counts()

    unpopular_products = product_review_count[product_review_count <= 15].index

    data.drop(data[data['asin'].isin(unpopular_products)].index, inplace=True)

    

    return data
# Textual Features

def extract_textual_features(data):

    

    # m: total number of training examples

    m = len(data['review'].values)

    

    # total number of characters

    text_length = np.array(data['review'].str.len()).reshape((m, 1))

    

    # total number of alphabetical characters

    character_count = np.array(

        data['review'].replace(regex=True, to_replace=r'[^a-zA-Z]', value=r'').str.len()

    ).reshape((m, 1))

    

    # Tokenized Sparse Matrix

    # vectorizer = CountVectorizer(lowercase=True)

    # matrix = vectorizer.fit_transform(np.array(data['review'].values))

    # matrix = np.array(matrix.todense())

    

    # word_count = np.sum(matrix, axis=1, keepdims=True)

    # unique_word_count = np.count_nonzero(matrix, axis=1).reshape((m, 1))

    

    # the number of words, unique words, and sentences in the review text

    sentence_count, word_count, unique_word_count = [], [], []

    for review in data['review'].values:

        # s = sent_tokenize(review)

        # sentence_count.append(len(s))

        text_blob = TextBlob(review)

        word_count.append(len(text_blob.words))

        unique_word_count.append(len(set(text_blob.words)))

        sentence_count.append(len(text_blob.sentences))

    word_count = np.array(word_count).reshape((m, 1))

    unique_word_count = np.array(unique_word_count).reshape((m, 1))

    sentence_count = np.array(sentence_count).reshape((m, 1))

    

    # Automated Readability Index

    ARI = 4.71 * (character_count / word_count) + 0.5 * (word_count / sentence_count)

    ARI = np.reshape(ARI, (m, 1))

    

    return np.concatenate(

        (text_length, character_count, word_count, unique_word_count, sentence_count, ARI),

        axis=1

    )
# Metadata Features

def extract_metadata_features(data):

    

    # m: total number of training examples

    m = len(data['overall'].values)

    

    # overall rating, the user gave to the product

    rating = np.array(data['overall'].values).reshape((m, 1))

    

    return rating
# Create bag of words

def create_bag_of_words(data):

    

    # Construct a bag of words matrix

    vectorizer = CountVectorizer(lowercase=True, stop_words="english", max_features=800)

    matrix = vectorizer.fit_transform(np.array(data['review'].values))

    

    return matrix.todense()



# Bag of Words matrix

# bow = create_bag_of_words(pd.concat([df_train, df_test]))
# Create tf-idf representation of the review text

def create_tf_idf_vector(data):

    

    # Construct a tf-idf matrix

    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

    matrix = vectorizer.fit_transform(data['review'].values)

    

    return matrix.todense()
# Create the feature vector and the result vector

def get_xy_vectors(data):

    

    # Constructing feature vector

    X = np.concatenate(

        (extract_textual_features(data), extract_metadata_features(data)),

        axis = 1

    )

    

    # m: total number of training examples

    m = X.shape[0]

    

    # Set the outcome variable (Denotes helpfulness of a review)

    upvotes = np.array(data['upvotes'].values).reshape((m, 1))

    downvotes = np.array(data['downvotes'].values).reshape((m, 1))

    Y = upvotes / (upvotes + downvotes)

    

    return X, Y
# Read from input file and store the contents in a dataframe

INPUT_FILE = '../input/reviews_Video_Games_5.json.gz'

df = parse_input(INPUT_FILE)
df.head()
# Remove unecessary data from the dataset

df = preprocess_input(df)

df = filter_input(df)

df.head()
# Separate data into training set and test set

df_train = df.sample(frac=0.8)

df_test = df.loc[~df.index.isin(df_train.index)]
# X_train: Input vector for the training set

# Y_train: Vector containing the results of the training set

X_train, Y_train = get_xy_vectors(df_train)



# X_test: Input vector for the test set

# Y_test: Vector containing the results of the test set

X_test, Y_test = get_xy_vectors(df_test)
# Bag of Words matrix

bow = create_bag_of_words(pd.concat([df_train, df_test]))
k = X_train.shape[0]



X_train1 = np.concatenate(

    (X_train, bow[:k, :]),

    axis=1

)



X_test1 = np.concatenate(

    (X_test, bow[k:, :]),

    axis=1

)
print('Dimensions of training set vectors:')

print('X: ', X_train1.shape)

print('Y: ', Y_train.shape)



print('\nDimensions of test set vectors:')

print('X: ', X_test1.shape)

print('Y: ', Y_test.shape)
# Vizualizing the distribution of helpfulness score across the data

plt.hist(Y_train, ec='black')

plt.xlabel('Helpfulness Score')

plt.show()
linear_model = LinearRegression(normalize=True)

linear_model.fit(X_train1, Y_train)
Y_pred = linear_model.predict(X_test1)
linear_model.score(X_test1, Y_test)