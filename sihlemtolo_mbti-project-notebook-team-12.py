# General libraries.

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import dill

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# Text mining libraries

import re

import urllib

import nltk

import string

from nltk.tokenize import word_tokenize, TreebankWordTokenizer

from nltk import LancasterStemmer, WordNetLemmatizer

from textblob import TextBlob, Word

from wordcloud import WordCloud, STOPWORDS

from numpy.linalg import svd



# Model building libraries

from imblearn.over_sampling import RandomOverSampler, SMOTE

import sklearn

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report, log_loss

from sklearn.feature_extraction.text import CountVectorizer



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from sklearn.model_selection import GridSearchCV



# Silencing warnings

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



import os

print(os.listdir("../input"))
# Loading the data.

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv('../input/test.csv')
# Defining the 'train' column.

df_train['train'] = 1

df_test['train'] = 0



# Now we will append them

df_all = df_train.append(df_test, sort=False)



# Displaying all dataframes.

display(df_train.head())

display(df_test.head())

display(df_all.head())



# Looking at the shapes of all dataframes.

print(df_train.shape)

print(df_test.shape)

print(df_all.shape)
# Checking for any NANs in the merged dataframe.

print(df_all['type'].isnull().sum(axis=0))

print(df_all['posts'].isnull().sum(axis=0))
def session_file(command, filename='notebook_env.db'):

    """Saves the notebook and provides a checkpoint for 

    when you need to exit the notebook whilst still 

    performing tasks.

    

    Arguments: 

    command - whether to save or reload from the last saved checkpoint. 

    Enter either 'save' or 'load'.

    filename - desired name of the saved file. If user wishes to 

    assign a custom name it should end in the extension '.db'.

    

    Output: a DB file saved to your folder.

    """

    if command == 'save':

        dill.dump_session(filename)

    else:

        # Runs if command is 'load'.

        dill.load_session(filename)
# Viewing the available personality types in our training set.

all_types = df_train['type'].unique().tolist()

all_types
# Plotting the distribution of personality types after seperating posts from each user.

x = df_train.type.value_counts()

plt.figure(figsize=(10,6))

ax = sns.barplot(x.index, x.values, alpha=0.8)

plt.title('The Distribution of the Different Personality Types')

plt.ylabel('Count')

plt.xlabel('MBTI Types')

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show();
def get_number_of_words(posts):

    # Seperating pipes from posts from each post.  

    parsed_posts = posts.split("|||")

    num_words = sum(len(post.split()) for post in parsed_posts)

    return num_words / 50

df_all['word_count'] = df_all['posts'].apply(get_number_of_words)
plt.figure(figsize=(15, 7))

ax = sns.distplot(df_all["word_count"], kde=True)

ax.set_title('Word Distributio')

ax.set_ylabel("Fraction of posts");

print(df_all["word_count"].mean())
#  Creating a column in the training set which has the sentiment score for each user.



df_train['polarity'] = df_train['posts'].map(lambda x: \

        TextBlob(x).sentiment.polarity)
x = round(df_train.groupby('type')['polarity'].mean(), 2)

plt.figure(figsize=(10,6))

ax = sns.barplot(x.index, x.values, alpha=0.8)

plt.title("MBTI_TYPES AVERAGE SEMTIMENT")

plt.ylabel('MEAN POLARITY')

plt.xlabel('MBTI_TYPES')

plt.show();
def generate_wordcloud(text, title):

    '''This piece of code was mostly

    taken from github. It creates

    word clouds'''

    # Create and generate a word cloud image:

    wordcloud = WordCloud(background_color="white").generate(text)



    # Display the generated image:

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(title, fontsize = 40)

    plt.show()
df_by_personality = df_all.groupby("type")['posts'].apply(' '.join).reset_index()

df_by_personality
for i, t in enumerate(df_by_personality['type']):

    text = df_by_personality.iloc[i,1]

    generate_wordcloud(text, t)
def treat_urls(

    df, 

    text='posts', 

    rename='no',

    delete_url='no'

    ):

    """This function performs specified operations on all the urls.

    

    Arguments:

    df - dataframe which user wants to perform operations on.

    text - column containing the text data.

    rename - renames all urls to 'url-web'.

    delete-urls - removes all the urls from the dataset.

    """

    urls = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    

    if rename == 'yes':

            

        subs_url = r'url-web'

        df[text] = df[text].replace(to_replace=urls, value=subs_url, regex=True) 

    if delete_url == 'yes':

        

        del_url = ' '

        df[text] = df[text].replace(to_replace=urls, value=del_url, regex=True)

    return df

df_all = treat_urls(df_all, rename='yes')

df_all.head()
def correct_spelling(df):

    '''This function corrects spelling errors present in the text.

    and outputs a dataframe with corrected spelling.

    '''

    df['posts'] = df['posts'].apply(lambda x: TextBlob(x).correct()) 

    return df

# Running the function and assigning it to a variable.

# df_all = correct_spelling(df_all)
def clean_post(post):

    ''' Converts the letters in the posts to lower case 

    and removes punctuation and digits'''



    # Convert all words to lower-case.

    post = post.lower()

    # Removing pipes.

    post = post.replace('|||', ' ')

    # Removing punctuation.

    post = re.sub('[%s]' % re.escape(string.punctuation), '', post)

    post = re.sub('\w*\d\w*', '', post)

    post = re.sub('[''""...@*#*]', '', post)

    return post

# Applying operations to the text column in the dataframe.

df_all['posts'] = df_all['posts'].apply(clean_post)

df_all.head()
def tokenize_lemmatize(df, text='posts'):

    '''Performs tokenisation and lemmatization on 

    the given text column.

    

    Arguments:

    df - dataframe which user wants to perform operations on.

    text - the column which operations should be performed on.

               

    Output:

    returns a dataframe with a two added columns.

    '''

    tokeniser = TreebankWordTokenizer()

    df['tokens'] = df[text].apply(tokeniser.tokenize)

    df['lemm'] = df['tokens'].apply(lambda x: ' '

                                    .join([Word(word).lemmatize()

                                           for word in x]))

    return df

  

# Tokenising and lemmatising the data using our custom function.

df_all = tokenize_lemmatize(df_all)

df_all.head()
def vectorize_data(df):

    '''Uses the CountVectorizer to converts our data from corpus to

    document term matrix'''

    vect = CountVectorizer(lowercase=True, stop_words='english')

    df_test = (df[df['train'] == 0])

    df_train = (df[df['train'] == 1])

    X_count_train = vect.fit_transform(df_train['posts'])

    X_count_test = vect.transform(df_test['posts'])

    return X_count_train, X_count_test



X, testing_X = vectorize_data(df_all)

y = df_train['type']
def data_preprocess(X,y):

    '''splits the data and returns two tuples'''

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,

                                                     random_state=42)

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = data_preprocess(X,y)
def resample_data(X, y):

    '''Resamples the data and returns two tuples for oversampled

    data and undersampled data'''

    sm = SMOTE()

    nm = NearMiss()

    X_sm,y_sm = sm.fit_sample(X, y)

    X_nm,y_nm = nm.fit_sample(X, y)

    return (X_sm, y_sm), (X_nm, y_nm)
def train_model(model, X_train, y_train):

    '''Fits a given model .

    '''

    mod = model()

    return mod.fit(X_train, y_train)
def predict(trained_model, X_test):

    '''Takes trained model and

    uses X_test to predict and

    returns the predicted y'''

    

    return trained_model.predict(X_test)

logreg_sm_model = train_model(LogisticRegression,

                              resample_data(X_train, y_train)[0][0],

                             resample_data(X_train, y_train)[0][1])

y_pred_logreg_sm_model = predict(logreg_sm_model, X_test)

logreg_nm_model = train_model(LogisticRegression,

                              resample_data(X_train, y_train)[1][0],

                             resample_data(X_train, y_train)[1][1])

y_pred_logreg_nm_model = predict(logreg_nm_model, X_test)
knn_sm_model = train_model(KNeighborsClassifier,

                              resample_data(X_train, y_train)[0][0],

                             resample_data(X_train, y_train)[0][1])



y_pred_knn_sm_model = predict(knn_sm_model, X_test)



knn_nm_model = train_model(KNeighborsClassifier,

                              resample_data(X_train, y_train)[1][0],

                             resample_data(X_train, y_train)[1][1])



y_pred_knn_nm_model = predict(knn_nm_model, X_test)
svc_sm_model = train_model(SVC,

                              resample_data(X_train, y_train)[0][0],

                             resample_data(X_train, y_train)[0][1])

y_pred_svc_sm_model = predict(svc_sm_model, X_test)



svc_nm_model = train_model(SVC,

                              resample_data(X_train, y_train)[0][0],

                             resample_data(X_train, y_train)[0][1])

y_pred_svc_nm_model = predict(svc_nm_model, X_test)

accuracy_svc_sm_model = metrics.accuracy_score(y_test,y_pred_svc_sm_model)

accuracy_svc_nm_model = metrics.accuracy_score(y_test,y_pred_svc_nm_model)

accuracy_logreg_sm_model = metrics.accuracy_score(y_test,y_pred_logreg_sm_model)

accuracy_logreg_nm_model = metrics.accuracy_score(y_test,y_pred_logreg_nm_model)



accuracy_knn_sm_model = metrics.accuracy_score(y_test,y_pred_knn_sm_model)

accuracy_knn_nm_model = metrics.accuracy_score(y_test,y_pred_knn_nm_model)
eval_dict = {'Model':['SVC SMOTE',

                       'SVC Near_miss',

                       'Logistic SMOTE',

                       'Logistic Near_miss',

                       'KNN SMOTE',

                       'KNN Near_miss'

                      ],

             'Accuracy': [accuracy_svc_sm_model,

                          accuracy_svc_nm_model,

                          accuracy_logreg_sm_model,

                          accuracy_logreg_nm_model,

                          accuracy_knn_sm_model,

                          accuracy_knn_nm_model

                         ]}



df_model_eval = pd.DataFrame(eval_dict)

# Let us now view our dataframe to see how each model does in terms accuracy.

df_model_eval
def param_tuning(X_train, y_train):

    """

    Applies GridSearchCV to

    find the best parameters for

    Logistic regression and

    returns the fitted 

    GridSearchCV object

    """

    

    # Setting possible C's and penalties that-

    #grid search must try

    params = [{'C':np.logspace(-3,3,10), 'penalty':['l1','l2']}]

    grid_search = GridSearchCV(estimator = LogisticRegression(),

                               param_grid = params,

                               scoring = 'accuracy',

                               cv = 3,

                               n_jobs = -1)

    grid_search = grid_search.fit(X_train, y_train)

    return grid_search



grid_search = param_tuning(X_train, y_train)

print("best params : ", grid_search.best_params_)

print("score : ", grid_search.best_score_)
# The numbers are added for the whole length of the dataframe

df_test['id'] = [i for i in range(1,len(df_test) + 1)]

# Our new id column is set as an index

df_test.set_index('id', inplace=True)
# generate predictions

prediction = logreg_sm_model.predict(testing_X)

# A new dataframe is made consisting of df_test id and predictions

df = pd.DataFrame({"id":df_test.index,"type":prediction})

df.set_index('id', inplace=True)
def encode_columns(df):

    '''

    Takes each personality type and split it into its four components

    and encodes them into 0 or 1

    '''

    df['mind'] = df['type'].str[0]

    df['energy'] = df['type'].str[1]

    df['nature'] = df['type'].str[2]

    df['tactics'] = df['type'].str[3]

    df.replace({"mind" : {"I":0,"E":1},

                 "energy":{"S":0,"N":1},

                 "nature":{"F":0, "T":1},

                 "tactics":{"P":0,"J":1}}, inplace = True)

    return df



df = encode_columns(df).drop('type', axis=1)
df.to_csv("submission.csv")
# Using our Dll function we defined earlier

session_file('save')