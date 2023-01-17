import pandas as pd

import numpy as np

import nltk

import re

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.tokenize import word_tokenize, TreebankWordTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

import string

from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

#from scipy.misc import imread

#conda install -c conda-forge wordcloud 

from wordcloud import WordCloud, STOPWORDS

#conda install -c conda-forge emoji

import emoji

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import metrics

from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

import warnings; warnings.simplefilter('ignore')
train = pd.read_csv('../input/train.csv') # Labelled Tweets (Training).

test = pd.read_csv('../input/test.csv') # Unlabelled Tweets (Predictions).

# Import a sample of the submission file .

sample = pd.read_csv('../input/random_example.csv')
# Create a list of Ids of the target variable

#creating label for type

train['type_id'] = train['type'].factorize()[0]

dic =dict(zip(train.type,train.type_id))

dic
# Inspect the training dataset.

# Theq independent variable 'posts' are associated with a particular personality type in the training dataset. 

# The feature 'type' refers to the target variable, which we will predict for the test dataset one we have fitted a model. 

train.head()
# Create a single dataset from the test and train tweets for preprocessing.

df_full = pd.concat([train,test],ignore_index=False,sort=True)

df_full = df_full[['posts']] # Exclude the target variable ('type')
# Inspect the concatenated dataset of tweets before preprocessing begins.

df_full.head()
def per_tweet(row):

    """ Create list of tweet posts by splitting on '|||' """

    l = []

    for i in row.split('|||'):

        l.append(len(i.split()))

    return np.var(l)

train['words_per_tweet'] = train['posts'].apply(lambda x: len(x.split())/50)

train['vocab_variance'] = train['posts'].apply(lambda x: per_tweet(x))

train.head()
plt.figure(figsize=(15,10))

sns.swarmplot("type", "words_per_tweet", data=train)

plt.title('Swarm plot of tweet length by personality type')
# Determine the number of tweets according to the personality type of the tweeter. 

train.groupby('type').agg({'type':'count'})
def plot_jointplot(mbti_type, axs, titles):

    ''' Joint-plot of vocab list against number of words per tweet for any of the mbti types'''

    

    df = train[train['type'] == mbti_type]

    sns.jointplot('vocab_variance', "words_per_tweet", data=df, kind="hex", ax = axs, title = titles)

    pass

## Generate joint-plots comparing lexicon and tweet length for each personality type.   

i = train['type'].unique()

k = 0

for m in range(0,2):

    for n in range(0,6):

        df = train[train['type'] == i[k]]

        sns.jointplot('vocab_variance', "words_per_tweet", data=df, kind="scatter")

        plt.title(i[k])

        k+=1
def WordCloud (train, ax, titles):

    fig, ax = plt.subplots(len(train['type'].unique()), sharex=True, figsize=(15,10*len(train['type'].unique())))

    '''Create word cloud for each personality type'''

    k = 0

    for i in train['type'].unique():

        df1 = train[train['type'] == i]

        wordcloud = WordCloud().generate(df1['posts'].to_string())

        ax[k].imshow(wordcloud)

        titles = ax[k].set_title(i)

        ax[k].axis("off")

        k+=1

    pass
def split_tweets(df,post = 'posts'):

    """ Splits posts delimmited by '|||' """

    

    df['posts'] = df['posts'].apply(lambda x: x.split('|||'))

   

    return df

df_full = split_tweets(df_full) 
def remove_urls(df):

    '''Define a string of characters to replace with the value.'''

    

    to_replace = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    value = r' url'

    

    df['posts'] = df['posts'].apply(lambda x:''.join(x))

    df['posts'] = df['posts'].replace(to_replace, value , regex = True)

    

    return df

# Use the replace function defined above to replace 

df_full = remove_urls(df_full)
def emoji_check(df):

    ''' Demojizes enables translation of emojicon characters  in to text within the tweets'''

    

    df['posts'] = df['posts'].apply(lambda x : ' '.join([emoji.demojize(i) if ((i in emoji.EMOJI_UNICODE) or (i in emoji.EMOJI_ALIAS_UNICODE)) else i for i in x.split()]))

    return df

# Remove emojicons, converting them to appropriate text where possible. 

df_full = emoji_check(df_full)
# Print list to see in-built string of punctuation. 

print(string.punctuation)
## Takes a while to run it 



def remove_punctuation(df):

    ''' Converts every character to lowercase, 

        buffer or replace certain letters with a white space''' 

    

    df['posts'] = df['posts'].str.lower()

    df['posts'] = df['posts'].apply(lambda x : ' '.join([i.replace('i’','i ').replace('.',' ').replace(':',' ') for i in x.split()]))

    df['posts'] = df['posts'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation + '1234567890']))

    

    return df



# Remove all remaining punctuation within the tweet text. 

df_full = remove_punctuation(df_full)


def tokenizing(df):

    ''' Parses text to remove individual words, called tokenization.''' 

    

    #df['tokens'] = df['post'].apply(lambda x : word_tokenize(''.join(x)))

    df_full['tokens'] = df_full['posts'].apply(TreebankWordTokenizer().tokenize)

    return df



df_full =tokenizing(df_full)
#Lemmatizing

#Set pos tag parameter from noun to verb. 

def lemma(df):

    

    ''' Loops through tokens and returns word to it's base form to reduce noise. '''

    lmtzr = WordNetLemmatizer()

    df['tokens']= df['tokens'].apply(lambda x : [lmtzr.lemmatize(i, pos='v') for i in x])

    #df['tokens'] = df['tokens'].apply(lambda x: [SnowballStemmer('english').stem(i) for i in x])

    return df

df_full = lemma(df_full)
df_full.head()
# Remove predefines stop words.

# Takes a long time to run. 

df_full['tokens'] = df_full['tokens'].apply(lambda x: [i for i in x if i not in set(stopwords.words('english'))])

# Create a column of the final set of features that will be used to train the models. 

df_full['features']=df_full['tokens'].apply(lambda x: ' '.join(x))
# Create list of words from tweets of the original training dataset.  

X_train_df = df_full.iloc[:len(train),2]

X_train_df
# Create test set of words

X_test_df = df_full.iloc[len(train):,2]

X_test_df
# Set the target variable as being the encoded personality types. 

y_train = train.type_id

y_train

def vectorize(X,x):

    '''The function takes two datasets, X for the train and x for test; 

    this will return scaled datasets'''

    

    tfidf_vect = TfidfVectorizer(min_df=0.001,

                                norm='l2',stop_words=set(stopwords.words('english')),

                             ngram_range=(1, 1),max_features=4000)

    X_vec = tfidf_vect.fit_transform(X).toarray()

    x_vec = tfidf_vect.transform(x).toarray()

    return (X_vec,x_vec)

X_train,X_test = vectorize(X_train_df,X_test_df)

X_test.shape
# Truncate sparse matrix to distill information. 

def truncate(X,x,c):

    '''This function takes in train, test and returns a subset number of features. '''

    length = len(X)

    a =pd.concat([pd.DataFrame(X),pd.DataFrame(x)],ignore_index=False)

    b = TruncatedSVD(n_components=c).fit_transform(a)

    X_train = b[:length,:]

    X_test = b[length:,:]

    return X_train, X_test

# Return a subset of 300 features. 

X_train, X_test = truncate(X_train,X_test,c=300)
X_test.shape


def parameter_search(X,y):

    '''This function will return the best estimators for our model'''

    

    parameters = {}

    count = 1

    for i in 'Mind Energy Nature Tactics'.split():

        parameters[i] = GridSearchCV(LogisticRegression(class_weight = 'balanced'),{'C':[0.001, 0.1,0.01,1,10],'max_iter':[50,100,150,500]}).fit(X_train,y.iloc[:,count])

        count += 1

    return parameters
def labels(train,y_train):

    '''fucntion returns the multi label to train our model.'''

    

    y=pd.DataFrame(train.type)

    

    dic = {'I':0,'E':1,'S':0,'N':1,'F':0,'T':1,'P':0,'J':1}

    count = 0

    for i in 'Mind Energy Nature Tactics'.split():

        y[i] = y['type'].apply(lambda x: dic[x[count]])

        count += 1

    return y

y = labels(train,y_train)
def models(X,y):

    '''This function will return a dictionary of models for each label.'''

    models ={}

    count = 1



    parameters = parameter_search(X_train,y) #this calls the previous fucntion from grid search

    for i in 'Mind Energy Nature Tactics'.split():

        models[i] = parameters[i].best_estimator_.fit(X_train,y.iloc[:,count])

        count += 1

    return models

models =models(X_train,y)


count=1

for i in 'Mind Energy Nature Tactics'.split():

    print(classification_report(y.iloc[:,count], models[i].predict(X_train)))

    print('****************************************************************')

    count +=1
def predictions(models,X_test):

    ''' This fucntion the full predictions for all four labels '''

    for i in 'Mind Energy Nature Tactics'.split():

        test[i] = models[i].predict(X_test)

    submit = test[['id','Mind','Energy','Nature','Tactics']]

    

    return submit

submit = predictions(models,X_test)



submit.to_csv('multi_label.csv',index=True)
y_train = train.type
def grid(X_train,y_train):

    '''This function will return the best estimators for our model'''

    lrc = GridSearchCV(LogisticRegression(class_weight='balanced'),{'C' : [0.001, 0.1,1, 10,100,1000],'max_iter':[50,100,150]}).fit(X_train,y_train)

    return lrc.best_estimator_

grid = grid(X_train,y_train)
def model(X,y,estimator):

    '''This function will return a dictionary of models for each label'''

    

    model = estimator.fit(X,y) 

    return model

model = model(X_train,y_train,grid)


'''this function call the previous one and print out the full report for all the classes'''



print(classification_report(y_train, model.predict(X_train)))

print('****************************************************************')
def predictions(X_test,test,model):

    ''' This fucntion the full predictions for the multi class '''

    y_pred = model.predict(X_test)

    test['type'] = y_pred

    

    dic = {'I':0,'E':1,'S':0,'N':1,'F':0,'T':1,'P':0,'J':1}

    count = 0

    for i in 'Mind Energy Nature Tactics'.split():

        test[i] = test['type'].apply(lambda x: dic[x[count]])

        count += 1

    return test

submit = predictions(X_test,test,model)[['id','Mind','Energy','Nature','Tactics']]

submit.to_csv('multi_class.csv',index=True)
sample.head()

def match(df,y_pred):

    

    ''' Uses the model.predict file as well as a dictionary of each of the personality traits 

    to assign the trait based on predicted probabilities. '''

    

    test['prediction'] = y_pred

    test['type'] = test['prediction'].apply(lambda x : dict(zip(train.type_id,train.type))[x])

    dic = {'I':0,'E':1,'S':0,'N':1,'F':0,'T':1,'P':0,'J':1}

    count = 0

    for i in 'Mind Energy Nature Tactics'.split():

        df[i] = df['type'].apply(lambda x: dic[x[count]])

        count += 1

    return df

#submit = match(test,y_pred)[['id','Mind','Energy','Nature','Tactics']]
X_train.shape
submit.head()

submit.to_csv('submit.csv',index=False)