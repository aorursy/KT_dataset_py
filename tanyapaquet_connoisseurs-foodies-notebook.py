import os

print(os.listdir("../input"))
# Import libraries for data manipulation and viewing

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# Load the provided training and test data into dataframes

train = pd.read_csv('../input/train.csv' )

test = pd.read_csv('../input/test.csv' )



# Check the dataframe sizes, type assignments, and missing values

print('Train info:')

train.info()

print('\nTest info:')

test.info()
# View the frequencies of the personality types

plt.subplots(figsize=(22, 5))

sns.countplot(train['type'].sort_values(), palette='Accent')
# Separate type categories

train['mind'] = train['type'].str[0]

train['energy'] = train['type'].str[1]

train['nature'] = train['type'].str[2]

train['tactics'] = train['type'].str[3]
# Encode category letters as either 0 or 1 

train['mind'] = train['mind'].apply(lambda x: 0 if x == 'I' else 1)

train['energy'] = train['energy'].apply(lambda x: 0 if x == 'S' else 1)

train['nature'] = train['nature'].apply(lambda x: 0 if x == 'F' else 1)

train['tactics'] = train['tactics'].apply(lambda x: 0 if x == 'P' else 1)



train.head()
import re



def word_replace(df):

    

    """Converts apostrophe suffixes to words, replace webpage links with url, annotate hashtags and mentions, remove a selection of punctuation, and convert all words to lower case.

    Args:

        df (DataFrame): dataframe containing 'posts' column to convert

    Returns:

        df (DataFrame): dataframe with converted 'posts' column 

    """



    # Change all webpage links to 'url'

    df['posts'] = df['posts'].str.replace(r'http.?://[^\s]+[\s]?', 'url ')

    

    # Replace apostrophe's with words

    df['posts'] = df['posts'].str.replace(r'n\'t', ' not')

    df['posts'] = df['posts'].str.replace(r'\'s', ' is')

    df['posts'] = df['posts'].str.replace(r'\'m', ' am')

    df['posts'] = df['posts'].str.replace(r'\'re', ' are')

    df['posts'] = df['posts'].str.replace(r'\'ve', ' have')

    df['posts'] = df['posts'].str.replace(r'\'ll', ' will')

    df['posts'] = df['posts'].str.replace(r'\'d', ' would')



    # Replace # and @ with word

    df['posts'] = df['posts'].str.replace(r'#|@', 'twithandle ')

    

    # Remove selected punctuation

    df['posts'] = df['posts'].str.replace(r"[',.():|-]", " ")



    # Convert all words to lower case

    df['posts'] = df['posts'].str.lower()

    

    return df
# Replace webpages, apostrophe's, selected punctuation etc.

train_clean = word_replace(train.copy())

test_clean = word_replace(test.copy())
# # Download NLTK libraries

# import nltk

# nltk.download()
# Import required nltk functions 

from nltk.tokenize import TweetTokenizer

from nltk.stem import WordNetLemmatizer



get_tokens = TweetTokenizer()

get_lemmas = WordNetLemmatizer()



# Define tokenization and lemmetization function

def tokenize_lemmatize(df):

    

    """Tokenize and lemmatize posts.

    Args:

        df (DataFrame): dataframe containing 'posts' column to convert

    Returns:

        df (DataFrame): dataframe with converted 'posts' column   

    """

    

    df['posts'] = df.apply(lambda row: [get_lemmas.lemmatize(w) for w in get_tokens.tokenize(row['posts'])], axis=1)  #Tokenize and lemmatize

    

    return df

  
# Tokenize and lemmatize posts

train_clean = tokenize_lemmatize(train_clean.copy())

test_clean = tokenize_lemmatize(test_clean.copy())
# # Install emoji library

# !pip install emoji
import emoji



# Create list of emoticon codes

emojies = set(emoji.UNICODE_EMOJI.keys())



# Define emoticon replacement functions

def swap_emoji(word):

    

    """Replace emoticon with 'emoji'.

    Args:

        word (str): word to replace

    Returns:

        word (str): replacement word   

    """

    

    if word in emojies:      #  replace emoticon

        return 'emoji'

    

    return word 





def emoji_convert(df):

    

    """Iterate through post and replace emoticons with 'emoji'.

    Args:

        df (DataFrame): dataframe containing 'posts' column to convert

    Returns:

        df (DataFrame): dataframe with converted 'posts' column   

    """ 

    

    df['posts'] = df['posts'].apply(lambda row: [swap_emoji(word) for word in row])   #  apply swap_emoji function to words in posts

    

    return df

# Substitute emoticon codes with 'emoji'

train_clean = emoji_convert(train_clean.copy())

test_clean = emoji_convert(test_clean.copy())
from nltk.corpus import stopwords



# List nltk English stopwords and add 'could' and 'would'

stop_words = stopwords.words('english')

stop_words = stop_words + ['could', 'would']



# Remove selected pronouns from stopword list

stop_words = [w for w in stop_words if w not in ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves']]



# Define function to remove stopwords

def remove_words(df):

    

    """Remove stopwords from posts.

    Args:

        df (DataFrame): dataframe containing 'posts' column to convert

    Returns:

        df (DataFrame): dataframe with converted 'posts' column   

    """ 

    

    df['posts'] = df['posts'].apply(lambda row: [word for word in row if word not in stop_words])   #iterate through words in posts and eliminate stopwords

    

    return df
# Remove stop words from tokens

train_clean = remove_words(train_clean.copy())

test_clean = remove_words(test_clean.copy())
# Create personality catagory and annotation lists

columns = ['mind', 'energy', 'nature', 'tactics']

ticks = [('I', 'E'), ('S', 'N'), ('F', 'T'), ('P', 'J')]



# Plot the number of posts per personality category

print('Number of posts per personality catagory:')

plt.subplots(figsize=(22, 100))

for i in range(len(columns)):

    j = i+1

    plt.subplot(22,4,j)

    ax = sns.countplot(train_clean[columns[i]], palette='Accent')

    ax.set_xticklabels(ticks[i])
# Calculate personality category class distributions

E = train_clean['mind'][train_clean['mind'] == 1].count() / train_clean['mind'].count()

print('E:I = {} : {}'.format(round(E, 2), 1-round(E, 2)))



N = train_clean['energy'][train_clean['energy'] == 1].count() / train_clean['energy'].count()

print('N:S = {} : {}'.format(round(N, 2), 1-round(N, 2)))



T = train_clean['nature'][train_clean['nature'] == 1].count() / train_clean['nature'].count()

print('T:F = {} : {}'.format(round(T, 2), 1-round(T, 2)))



J = train_clean['tactics'][train_clean['tactics'] == 1].count() / train_clean['tactics'].count()

print('J:P = {} : {}'.format(round(J, 2), 1-round(J, 2)))
# Count the number of emoticons used

train_clean['emoji_count'] = train_clean['posts'].apply(lambda row: len(['emoji' for word in row if word == 'emoji']))



# Plot the average number of emoticons used per personality category

print('Average number of emoticons used per personality category:')

plt.subplots(figsize=(22, 100))

for i in range(len(columns)):

    j = i+1

    plt.subplot(22,4,j)

    df = train_clean.groupby(columns[i])[['emoji_count']].mean()

    df.index = ticks[i]

    sns.barplot(x=df.index, y=df['emoji_count'], palette='Greens')

    plt.xlabel(columns[i])
# Count the number of webpage links

train_clean['url_count'] = train_clean['posts'].apply(lambda row: len(['url' for word in row if word == 'url']))



# Plot the average number of webpage links per personality category

print('Average number of webpage links per personality category:')

plt.subplots(figsize=(22, 100))

for i in range(len(columns)):

    j = i+1

    plt.subplot(22,4,j)

    df = train_clean.groupby(columns[i])[['url_count']].mean()

    df.index = ticks[i]

    sns.barplot(x=df.index, y=df['url_count'], palette='Blues')

    plt.xlabel(columns[i])
# Count the number of exclamation marks used

train_clean['exclamation'] = train_clean['posts'].apply(lambda row: len(['!' for word in row if word == '!']))



# Plot the average number of exclamation marks used per personality category

print('Average number of exclamation marks used per personality category:')

plt.subplots(figsize=(22, 100))

for i in range(len(columns)):

    j = i+1

    plt.subplot(22,4,j)

    df = train_clean.groupby(columns[i])[['exclamation']].mean()

    df.index = ticks[i]

    sns.barplot(x=df.index, y=df['exclamation'], palette='Purples')

    plt.xlabel(columns[i])
# Count the use of pronoun 'I'

train_clean['I'] = train_clean['posts'].apply(lambda row: len(['i' for word in row if word == 'i']))



# Plot the average number of times 'I' is used per personality category

print("Average use of 'I' per personality category:")

plt.subplots(figsize=(22, 100))

for i in range(len(columns)):

    j = i+1

    plt.subplot(22,4,j)

    df = train_clean.groupby(columns[i])[['I']].mean()

    df.index = ticks[i]

    sns.barplot(x=df.index, y=df['I'], palette='Oranges')

    plt.xlabel(columns[i])
# Count the use of twitter handles

train_clean['handle'] = train_clean['posts'].apply(lambda row: len(['handle' for word in row if word == 'twithandle']))



# Plot the average number of twitter handles used per personality category

print("Average use of twitter handles per personality category:")

plt.subplots(figsize=(22, 100))

for i in range(len(columns)):

    j = i+1

    plt.subplot(22,4,j)

    df = train_clean.groupby(columns[i])[['handle']].mean()

    df.index = ticks[i]

    sns.barplot(x=df.index, y=df['handle'], palette='Reds')

    plt.xlabel(columns[i])
# Import vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



# Initialise vectorizer

tt = TfidfVectorizer(preprocessor=list, tokenizer=list, ngram_range=(1,2), min_df=2, smooth_idf=False)



# Define vectorization function

def vectorise(train_set, test_set):

    

    """Fit a vector to train_set and transform train_set and test_set accordingly. 

    Args:

        train_set (array or DataFrame): train features to vectorize

        test_set (array or DataFrame): test features to vectorize

    Returns:

        train_vect (sparse array): train_set vector

        test_vect (sparse array): test_set vector

    """

    

    tt.fit(train_set)

    train_vect = tt.transform(train_set)

    test_vect = tt.transform(test_set)

    

    return train_vect, test_vect

# Import models, metrics, and model selection utilities

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import log_loss, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
# Vectorize training posts

X_train_vect, X_test_vect = vectorise(train_clean['posts'], test_clean['posts'])

X_train_vect.shape
# Initialize the model to be used

cm = LogisticRegression(penalty='l1', solver='liblinear', random_state=11)



# Specify the range of 'C' parameters

params = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}



# Initialize the grid search

grid = GridSearchCV(cm, param_grid=params, scoring='neg_log_loss', n_jobs=-1, cv=4)



# Execute a grid search for each personality category

categories = ['mind', 'energy', 'nature', 'tactics']

for cat in categories:

    grid.fit(X_train_vect, train_clean[cat])

    print(cat + ':')

    print('Best score: ', -grid.best_score_)

    print('Best paramaters: ', grid.best_params_)

    print('\n')
# Partition a rondom test set from the training data

X_train, X_test, y_train, y_test = train_test_split(train_clean['posts'], train_clean[['mind', 'energy', 'nature', 'tactics']], random_state=11)



# Vectorize the training and test sets

X_train_vect, X_test_vect = vectorise(X_train, X_test)

X_train_vect.shape
# Define model building function

def build_model(X_train_vect_f, X_test_vect_f, y_train_f, y_test_f, model):

    """Fit the specified model, make predictions, and calculate the log loss and confusion matrix.

    Args:

        X (array or DataFrame): independent variables

        y (array or DataFrame): dependent variable

        model (model): sklearn model and parameters

    Returns:

        train_pred (array): predictions from training data 

        test_pred (array): predictions from test data

        train_loss (float): log loss on training data

        test_loss (float): log loss on test data

        test_accuracy (float): accuracy score on test data

        test_matrix (array): confusion matrix on test data 

    """

    

    model.fit(X_train_vect_f, y_train_f)                         # fit model using training data



    train_pred = model.predict(X_train_vect_f)                   # predict training set labels

    test_pred = model.predict(X_test_vect_f)                     # predict test set labels

    

    train_proba = model.predict_proba(X_train_vect_f)            # predict label probabilities from training set

    test_proba = model.predict_proba(X_test_vect_f)              # predict label probabilities from test set

    

    train_loss = log_loss(y_train_f, train_proba, eps=1e-15)     # log loss on training data

    test_loss = log_loss(y_test_f, test_proba, eps=1e-15)        # log loss on test data   

    

    test_accuracy = accuracy_score(y_test_f, test_pred)          # accuracy score on test data

    test_matrix = confusion_matrix(y_test_f, test_pred)          # confusion matrix on test data

        

    return train_pred, test_pred, train_loss, test_loss, test_accuracy, test_matrix
# Initialize a list to store log loss and accuracy on test sets

ll_scores = []

ac_scores = []
# Initialize model for the mind category

cm = LogisticRegression(penalty='l1', solver='liblinear', C=4, random_state=11)



# Train and test the model for the mind category

pred_train, pred_test, loss_train, loss_test, acc_test, matrix_test = build_model(X_train_vect, X_test_vect, y_train['mind'], y_test['mind'], cm)

ll_scores.append(loss_test)

ac_scores.append(acc_test)

print('Mind:')

print('Log loss on training set: ',loss_train)

print('Log loss on test set: ', loss_test)

print('Accuracy on test set:', acc_test)

print('Test set confusion matrix: \n', matrix_test)
# Initialize model for the energy category

cm = LogisticRegression(penalty='l1', solver='liblinear', C=5, random_state=11)



# Train and test the model for the energy category

pred_train, pred_test, loss_train, loss_test, acc_test, matrix_test = build_model(X_train_vect, X_test_vect, y_train['energy'], y_test['energy'], cm)

ll_scores.append(loss_test)

ac_scores.append(acc_test)

print('Energy:')

print('Log loss on training set: ',loss_train)

print('Log loss on test set: ', loss_test)

print('Accuracy on test set:', acc_test)

print('Test set confusion matrix: \n', matrix_test)
# Initialize model for the nature category

cm = LogisticRegression(penalty='l1', solver='liblinear', C=5, random_state=11)



# Train and test the model for the nature category

pred_train, pred_test, loss_train, loss_test, acc_test, matrix_test = build_model(X_train_vect, X_test_vect, y_train['nature'], y_test['nature'], cm)

ll_scores.append(loss_test)

ac_scores.append(acc_test)

print('Nature:')

print('Log loss on training set: ',loss_train)

print('Log loss on test set: ', loss_test)

print('Accuracy on test set:', acc_test)

print('Test set confusion matrix: \n', matrix_test)
# Initialize model for the tactics category

cm = LogisticRegression(penalty='l1', solver='liblinear', C=3, random_state=11)



# Train and test the model for the tactics category

pred_train, pred_test, loss_train, loss_test, acc_test, matrix_test = build_model(X_train_vect, X_test_vect, y_train['tactics'], y_test['tactics'], cm)

ll_scores.append(loss_test)

ac_scores.append(acc_test)

print('Tactics:')

print('Log loss on training set: ',loss_train)

print('Log loss on test set: ', loss_test)

print('Accuracy on test set:', acc_test)

print('Test set confusion matrix: \n', matrix_test)
print('Average log loss across personality categories: ', round(np.mean(ll_scores), 3))

print('Average accuracy across personality categories: ', round(np.mean(ac_scores), 3))
# Initialize output dataframe

results = test[['id']]



# Vectorize the training and test sets

X_train_vect, X_test_vect = vectorise(train_clean['posts'], test_clean['posts'])

X_train_vect.shape
c_vals = [4, 5, 5, 3]      # C parameter list for final models

class1_proba = []          # List to store model probability predictions



# Train final classification models for each category

for i in range(4):

    cm = LogisticRegression(penalty='l1', solver='liblinear', C=c_vals[i], random_state=11)

    cm.fit(X_train_vect, train[categories[i]].values)

    pred_test = cm.predict(X_test_vect)                 # Predict labels

    results[categories[i]] = pred_test                  # Compile label results

    probability = cm.predict_proba(X_test_vect)         # Predict probabilities

    class1_proba.append(probability[:,1])               # Save probabilities for predicting class label 1
results.to_csv('MBTI_l1_liblinear_C4553_split_tt_all_pp_emoji_lib_handles.csv', index=False)

# Kaggle public score: 4.64397
# Compile probability results

for i in range(4):

    results[categories[i]] = class1_proba[i]
results.to_csv('MBTI_l1_liblinear_C4553_split_tt_all_pp_emoji_lib_handles_proba.csv', index=False)

# Kaggle public score: 0.34227