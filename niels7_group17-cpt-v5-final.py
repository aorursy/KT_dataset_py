import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import TreebankWordTokenizer

import re

import string
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
pd.options.display.max_columns = 1000

pd.options.display.max_rows = 1000
train = pd.read_csv('../input/train.csv', encoding='utf-8')

test = pd.read_csv('../input/test.csv', encoding='utf-8', index_col='id')

example = pd.read_csv('../input/random_example.csv', encoding='utf-8', index_col='id')
# See how big the data sets are.

print('Training data size: ', train.shape[0])

print('Test data size: ', test.shape[0])
# Create and set ids for train to continue from test ids.

train = train.reset_index()

train = train.rename(columns={'index': 'id'})

train['id'] = train['id']+test.shape[0]+1

test['type'] = '????'
# Combine the test and train sets.

combined = pd.concat([test.reset_index(), train], sort=False)

combined.set_index('id', inplace=True)
# Check for any missing values.

combined.isna().sum()
types = list(train['type'].unique())

print('The 16 MBTI types are: ' + str(types))
# Create the lists.

labels = ['mind','energy','nature','tactics']

label_letters = ['E','N','T','J']

alt_label_letters = ['I', 'S', 'F', 'P']
# Adds a column for each of the four MBTI function labels.

# Maps each letter of the personality type into a binary labeled cognitive function.





def convert_type_to_int(df):

    '''

    Adds a column for each of the four MBTI categories.

    Assigns a binary value based on the personality type column.

    

    Parameters

    ----------

    df : Pandas dataframe that includes the type column to be

    encoded.

    

    Returns

    -------

    df : dataframe

    Dataframe containing four new columns with binary values.

        

    Examples

    --------

    df = convert_type_int(df)

    '''

    for i in range(len(labels)):

        df[labels[i]] = df['type'].apply\

        (lambda x: x[i] == label_letters[i]).astype('int')

    return df
data = combined
data = convert_type_to_int(data)
data.shape
data.head()
data.tail()
# Plot the number of times each personality type is observed in the train set.

train_count = train['type'].value_counts()



_ = plt.figure(figsize=(15, 5))

_ = sns.barplot(train_count.index, train_count.values, alpha=0.8)

_ = plt.ylabel('Number of Observations', fontsize=14)

_ = plt.xlabel('Personality Type', fontsize=14)

_ = plt.title('Count of the Training Set MBTI Observations', fontsize=18)
# Plot count bar graphs for each MBTI function.

plt.figure(figsize=(15, 10))

for i, label in enumerate(labels):

    ax = plt.subplot(2, 2, i+1)

    label_count = convert_type_to_int(train)[label].value_counts()

    g = sns.barplot(label_count.index, label_count.values, alpha=0.8)

    plt.ylabel('Number of Observations', fontsize=14)

    plt.title(label, fontsize=14)

    g.set(xticklabels=[alt_label_letters[i], label_letters[i]])
# Splits every post on the '|||' string.

def separate_posts(post):

    return ' '.join(post.split('|||'))
# Use regular expressions to remove website urls.

def remove_urls(post):

    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    subs_url = r' '

    return re.sub(pattern_url, subs_url, post) 
# Remove numbers from text.

def remove_numbers(post):

    p_numbers = '0123456789'

    return ''.join([l for l in post if l not in p_numbers])
# Remove punctuation from text.

def remove_punctuation(post):

    return ''.join([l for l in post if l not in string.punctuation])
# Remove apostrophes that are missed in normal punctuation.

def remove_strange(post):

    return post.replace("‘", '').replace("’", '').replace("'", '')
# Convert all text to lower case.

def lower_case(post):

    return post.lower()
# Remove extra white spaces and leave only single white spaces between words.

def remove_extra_spaces(post):

    return re.sub('\s+', ' ', post)
# Break the post string up into words (tokens)

def tokenizer_func(post):

    return TreebankWordTokenizer().tokenize(post)
# Create a custom list of stopwords to test.

# The set excludes personal pronouns.

custom_stop_words = [remove_punctuation(word) for word in stopwords.words('english')[39:]]
# Remove stop words.

def remove_stop_words(tokens, stop_words=custom_stop_words):

    return [token for token in tokens if (token not in stop_words) and (len(token)<15)]
# Lemmatize the words to a recognised English base form.

def lem_func(words, lemma = WordNetLemmatizer()):

    return [lemma.lemmatize(word) for word in words if word not in custom_stop_words]
# Join invividual tokens into a single string.

def join_tokens(post):

    return ' '.join(post)
# Main function that calls the individual selected cleaning steps.

def clean_data(df, col, cleaning_funcs):

    '''

    Separate each observation into a string of posts.

    Applies a list of text cleaning functions to the

    specified column 'col'.

    

    Parameters

    ----------

    df : Pandas dataframe that includes the data column to 

    be cleaned.

    

    col : string

    The column name that contains the data to be cleaned.

    

    cleaning_funcs : list

    List of functions to be applied to the dataframe column.

    

    Returns

    -------

    df : dataframe

    Dataframe containing a new column with the cleaned data.

    

    Examples

    --------

    df = cleaned_data(df, 'posts', cleaning_funcs)

    '''

    df['posts_processed'] = df[col].apply(separate_posts)

    

    for func in cleaning_funcs:

        df['posts_processed'] = df['posts_processed'].apply(func)

    

    return df
# Make naive predictions.

for label in labels:

    train['naive_'+label] = train[label].value_counts().index[0]
# Create a results table for the naive guesses.

naive_results = pd.DataFrame(data=[], index = ['Naive_Accuracy'], columns = labels)

for label in labels:

    naive_results[label] = metrics.accuracy_score(train[label], train['naive_'+label])
naive_results
# Print the naive confusion matrices.

for label in labels:

    print('\nConfusion matrix for ' + label + ' is:')

    print(metrics.confusion_matrix(train[label], train['naive_'+label]))
'''Create a list of which cleaning steps to use.

This list contains all the steps in the correct order.

Careful consideration must be applied when using a 

subset of this list.'''

cleaning_funcs = [remove_urls,

                  remove_numbers,

                  remove_punctuation,

                  remove_strange,

                  lower_case,

                  remove_extra_spaces,

                  tokenizer_func,

                  remove_stop_words,

                  lem_func,

                  join_tokens]
data = clean_data(data, 'posts', cleaning_funcs)
# Create an instance of a counting vectorizer.

# Use the default parameters for first test.

countV = CountVectorizer()

# Transform the cleaned data into vectors.

X_countV = countV.fit_transform(data['posts_processed'])
X_countV.shape
# Create an instance of a Tfidf vectorizer.

# Use the default parameters for first test.

tfidf = TfidfVectorizer()

# Transform the cleaned data into vectors.

X_tfidfV = tfidf.fit_transform(data['posts_processed'])
X_tfidfV.shape
# Remove the unlabeled test/submission portion of the data from the data to be used for training.

X1 = X_countV[len(test):]

X2 = X_tfidfV[len(test):]

y = data[len(test):]
# Evaluate the model performance.





def evaluate(model, X, y):

    '''

    Evaluates the model performance for a given X and y sets.

    Splits the data into a training and test set.

    Fits the model to the training set.

    Calculates the accuracy, precision, recall and F1 scores of the test set.

    Calculates the averages of the scores.

    Prints the results dataframe.

    

    Parameters

    ----------

    model : sklearn classifier.

    

    X : list of sparse matrices

    The first sparse matrix must be a countvectorized matrix or 0.

    The second sparse matrix must be a Tfidf matrix or 0.

    

    y : panadas series

    Series containing the training labels for the data in X.

    

    Returns

    -------

    None

    

    Examples

    --------

    evaluate(logR, [X1,X2], y)

    evaluate(knn, [0,X2], y)

    '''

    # Perform evaluation for all datasets passed.

    for i, Xi in enumerate(X):

        results = pd.DataFrame(data=[], index = ['Accuracy', 'Precision', 'Recall'], columns = labels)

        x_vect = ['Count', 'TFIDF']

        if type(Xi) is not int:

            print('Results for ', x_vect[i])

            # Perform the model fit steps for each label in the MBTI functions labels.

            for label in labels:

                X_train, X_test, y_train, y_test = train_test_split(Xi,

                                                                    y[label],

                                                                    test_size=0.2,

                                                                    random_state=42)

                model.fit(X_train,y_train)

                y_test_pred = model.predict(X_test)

                results.loc['Accuracy', label] = metrics.accuracy_score(y_test, y_test_pred)

                results.loc['Precision', label] = metrics.precision_score(y_test, y_test_pred)

                results.loc['Recall', label] = metrics.recall_score(y_test, y_test_pred)

                results.loc['F1 test', label] = metrics.f1_score(y_test, y_test_pred)

            results['average'] = results.mean(axis=1)

            print(results)
# Instantiate a model and call the evaluate function.

logR = LogisticRegression(solver='lbfgs',n_jobs=-1)

evaluate(logR, [X1,X2], y)
# Instantiate a model and call the evaluate function.

logR = LogisticRegression(class_weight='balanced', solver='lbfgs', n_jobs=-1)

evaluate(logR, [X1, X2], y)
naive_results
# Instantiate and evaluate the model.

knn = KNeighborsClassifier()

evaluate(knn, [0, X2], y)
# Output the naive guesses

naive_results
# Create an instance of a word importance vectorizer.

tfidf = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.9)

# Transform the cleaned data into vectors.

X_tfidfV = tfidf.fit_transform(data['posts_processed'])
X3 = X_tfidfV[len(test):]
ks = [3, 5, 10]

param_grid = {'n_neighbors': ks}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='f1', cv=5, return_train_score=False)
# Fit the gridsearch function for the knn model to the X and y data.

grid_knn.fit(X3, y[labels[0]])
# Create a dataframe from the grid search results.

pd.DataFrame(grid_knn.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]
randF = RandomForestClassifier(n_estimators=10)

evaluate(randF, [0, X3], y)
naive_results
# Create a list of which cleaning steps to use.

# This list contains all the steps in the correct order.

# A subset of this list can be used.

# The order must remain the same.

cleaning_funcs = [remove_urls,

                  remove_numbers,

                  remove_extra_spaces]
data = clean_data(data, 'posts', cleaning_funcs)
# Create an instance of a word importance vectorizer.

tfidf = TfidfVectorizer(stop_words='english',

                        max_df=0.8,

                        min_df=2)

# Transform the cleaned data into vectors.

X_tfidfV = tfidf.fit_transform(data['posts_processed'])
X_tfidfV.shape
X4 = X_tfidfV[len(test):]

X_sub = X_tfidfV[:len(test)]

y = data[len(test):]
# Instantiate the logistic model.

logR = LogisticRegression(class_weight='balanced', max_iter=1000)



# Create the parameters grid dictionary.

logR_parameters = {'C': [0.01, 0.1, 1.0, 10],

                   'solver': ('liblinear','saga','lbfgs')

                  }



# Set up the grid search.

logR_grid = GridSearchCV(estimator=logR,

                         param_grid=logR_parameters,

                         cv=5,

                         return_train_score=False,)
sub = example
# Loop through every MBTI category label.

# Fit the model to predict the specific label.

# Find and use the optimal hyperparameters.

for label in labels:

    y_train = y[label]

    X_train = X4



    logR_grid.fit(X_train, y_train)

    

    sub[label] = logR_grid.predict(X_sub)

    

    results = pd.DataFrame(logR_grid.cv_results_)

    print(results[results['rank_test_score']==1][['params', 'mean_test_score']])
# Check if the output format is correct.

sub.head()
# Create a csv output file with the predictions.

sub.to_csv('submission.csv')