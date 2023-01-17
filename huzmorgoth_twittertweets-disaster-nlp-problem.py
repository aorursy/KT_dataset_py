# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# import datasets

trDS = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

tsDS = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# Extract dataset shapes

trShape = trDS.shape

tsShape = tsDS.shape

print(trShape, tsShape)
# merge datasets for preprocessing



tsDS['target'] = 1

DS = []

DS.append(trDS)

DS.append(tsDS)



DS = pd.concat(DS, axis = 0)



DS.shape
# Dataset info

DS.info()
# print first 5 rows

DS.head()
# print columns and their null values

DS.isnull().sum()
# Visualising missing values in the DS

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(DS.isna().columns, DS.isna().sum())

ax.set_xlabel("Features")

ax.set_ylabel("Counts")

ax.set_title("Number of missing values in the Dataset")

plt.show()
# First we shall deal with the Keyword feature

# Top 20 keywords in the DS



print(f"Total number of the unique keywords in Dataset: {DS.keyword.nunique()}")



fig, ax = plt.subplots(figsize=(15,5))



DS.keyword.value_counts()[:25].plot(kind="bar")

ax.set_title("Top 25 unique keywords in the Dataset")



plt.show()
# functions for extracting the probality of the event being a disaster, with the present keywords

def disaster_prob(df):

    """

    Imputes the empty values in the dataframe with "none" and groups df by keyword column.

    Displays probability of a disaster when a keyword is used and the amount of times that word is listed in keywords.



    """

    x = df.copy()

    x.fillna("none")

    

    key_prob = x.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(

        columns={'text':'Count', 'target':'Disaster Probability'})



    return key_prob
key_prob = disaster_prob(DS)
# Grouping the probablity by keywrods and is more than 0.5

key_prob[key_prob["Disaster Probability"] > 0.5].sort_values("Disaster Probability", ascending=False)
# As noticed, the missing values are not in the above mentioned list or the probability them is under 0.5 

# which indicates that thenone category belong the other class with higher probability

# we can simply replace missing keywords with "None"

DS.keyword = DS.keyword.fillna('none')
# Dealing with location missing values

# Top 25 locations in the Datset

print(f"Total unique locations in the Dataset: {DS.location.nunique()}")



fig, ax = plt.subplots(figsize=(15,5))



DS.location.value_counts()[:25].plot(kind="bar")

ax.set_title("Top 25 unique Locations in the Dataset")



plt.show()

# we could replace the similar location values with a common term but

# it won't make much of a difference and we could replace the NaN value with the location 'worldwide' or any new term 

# but due to higher count of missing values we will only be giving importance to the new term and confuse the model

# we shall replace the nan values with "" empty values, as we are going to combine all the features for the final corpus

DS.location = DS.location.fillna('')
DS.isnull().sum()
"""

--------------------------------------

Visualising and preprocess the Train text through NLP piplines

--------------------------------------

"""
# Splitting the Train and Test again

x = trShape[0]

Train = DS.iloc[:x]

Test = DS.iloc[x:]



print(Train.shape, Test.shape)
# distribution of target

print(Train.target.value_counts())



def TragetDist(Train):

    disaster ={"1": sum(Train[Train.target == 1].target.value_counts())}

    not_disaster = {"0": sum(Train[Train.target == 0].target.value_counts())}



    fig, ax = plt.subplots(figsize=(8,5))



    ax.bar(not_disaster.keys(), not_disaster.values(), label="Not disaster")

    ax.bar(disaster.keys(), disaster.values(), label="Disaster")

    plt.title("Target distribution")

    plt.xlabel("Category")

    plt.ylabel("Number of samples")

    ax.legend()



    plt.show()
TragetDist(Train)
# As shown in the graph, Non-disaster class examples are dominating the Train DS

# Therefore, downsampling the Majority class rows

from sklearn.utils import resample



df_majority = Train[Train.target==0]

df_minority = Train[Train.target==1]

 

# Down-sample majority class

df_majority_downsample = resample(df_majority, replace=False,     # sample with replacement

                                  n_samples=3271,    # to match minority class

                                  random_state=123) # reproducible results

 

# Combine majority class with Down-sample majority class

df_dsampled = pd.concat([df_majority_downsample, df_minority])
print(df_dsampled.target.value_counts())

TragetDist(df_dsampled)
# words per tweet

def word_per_tweet_vis(Train):

    disaster_uniques = Train[Train.target==1].text.str.split().map(lambda x: len(x))

    non_disaster_uniques = Train[Train.target==0].text.str.split().map(lambda x: len(x))



    fig, axs = plt.subplots(1,2, figsize=(15,5))

    axs[0].hist(non_disaster_uniques, color="blue")

    axs[1].hist(disaster_uniques, color="red")



    axs[0].set_title("Not Disaster")

    axs[1].set_title("Disaster")

    fig.suptitle("Words per tweet")



    plt.show()

    

# average length per word

def avgLen_word_vis(Train):

    disaster_word_len = Train[Train.target==1].text.str.split().apply(lambda x : [len(i) for i in x])

    non_disaster_word_len = Train[Train.target==0].text.str.split().apply(lambda x : [len(i) for i in x])



    fig, axs = plt.subplots(1,2, figsize=(15,5))

    sns.distplot(non_disaster_word_len.map(lambda x: np.mean(x)), ax=axs[0], color="blue")

    sns.distplot(disaster_word_len.map(lambda x: np.mean(x)), ax=axs[1], color="red")



    axs[0].set_title("Not Disaster")

    axs[1].set_title("Disaster")

    fig.suptitle("Average word length per tweet")



    plt.show()

    

# characters per tweet

def char_per_tweet_vis(Train):

    disaster_len = Train[Train.target==1].text.str.len()

    not_disaster_len = Train[Train.target==0].text.str.len()



    fig, axs = plt.subplots(1,2, figsize=(15,5))

    axs[0].hist(not_disaster_len, color="blue")

    axs[1].hist(disaster_len, color="red")



    axs[0].set_title("Not Disaster")

    axs[1].set_title("Disaster")

    fig.suptitle("Characters per tweet")



    plt.show()
# Number of words per tweet

word_per_tweet_vis(Train)
# Avg length per word

avgLen_word_vis(Train)
# Number of characters per tweet

char_per_tweet_vis(Train)
# import required libraries for NLP

import re

import string

import nltk

from nltk import word_tokenize

from nltk.tokenize import TweetTokenizer

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk import FreqDist

from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from nltk import flatten
nltk.download('stopwords')
def create_corpus(df, target=None, var=str, full_corpus=False): 

    """

    Performs TweetTokenizer() from nltk and creates a corpus based on the distinct target group.

    When full_corpus=True or when target==None then a corpus without the

    distinct target group would be created.

    

    Parameters

    ----------

    df: DataFrame that contains text 

    var: The variable that contains the text to be processed

    target: The distinct category group should be created based on

    test: bool, when true a corpus for test set will be created

    

    Returns

    -------

    A corpus as a list

    """

    x = df.copy()

    

    # removing html

    x["text"] = x["text"].apply(lambda y: re.split(r"http\S+", str(y))[0])

    

    if full_corpus==True or target==None:

        corpus = flatten([TweetTokenizer(strip_handles=True).tokenize(i.lower()) for i in x[var].tolist()])

    else: 

        corpus = flatten([TweetTokenizer(strip_handles=True).tokenize(i.lower()) for i in x[x["target"]==target].text])

    return corpus



def extract_punctuation(corpus):

    """

    Extracts punctuation from the corpus.

    

    Parameters

    ----------

    corpus: list of tokens

    

    Returns

    -------

    A sorted dictionary which has the keys as the unique special character

    and the values is the amount of times that distinct key occurs within the corpus. 

    """

    dictionary = defaultdict(int)

    for i in corpus:

        if i in string.punctuation:

            dictionary[i] += 1

    return dictionary





def plot_dict_freq(dictionary):

    """

    Visualization of keys and their frequency.

    

    Parameters

    ----------

    dictionary: A dictionary type

    

    Returns

    -------

    The visualization of the frequency with respect to the unique key.

    """

    x, y = zip(*dictionary.items())

    

    fig, ax = plt.subplots(figsize=(15,5))

    plt.bar(x, y)

    plt.ylabel("Frequency")



    return plt.show()
"""

Creating corpus for visualisation from the Text variable of the Train DS.



"""


# creating a corpus of words based on the target 

disaster_corpus = create_corpus(df=Train, var="text", target=1)

non_disaster_corpus = create_corpus(df=Train, var="text", target=0)



# corpus of full train

train_corpus = create_corpus(df=Train, var="text", full_corpus=True)



# extracting punctuation counts

disaster_corpus_punc = extract_punctuation(disaster_corpus)

non_disaster_corpus_punc = extract_punctuation(non_disaster_corpus)



full_train_corpus_punc = extract_punctuation(train_corpus)
# visualising the frequency of a special character in non-disasterous tweets

plot_dict_freq(non_disaster_corpus_punc)
# visualizing the frequency of a special character in disasterous tweets

plot_dict_freq(disaster_corpus_punc)
# visualizing the frequency of a special character in full train set corpus

plot_dict_freq(full_train_corpus_punc)


def get_top_n_words(corpus, ngrams=(1,1), n=None):

    """

    List the top n ngrams in corpus according to counts.

    

    Parameters

    ----------

    corpus: Text to be counted in Bag of Words

    ngrams: Amount of n words. ngrams=(min, max)

    n: Number of n words to display

    

    Returns

    -------

    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 

    [('python', 2),



('world', 2),

     ('love', 2),

     ('hello', 1),

     ('is', 1),

     ('programming', 1),

     ('the', 1),

     ('language', 1)]

    """

    vec = CountVectorizer(ngram_range=ngrams).fit(corpus)

    bow = vec.transform(corpus)

    sum_words = bow.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
stp = stopwords.words('english')

stp_wrds_ = set(stp)
# Top 15 unigrams when there is a disaster tweet

# remove stopwods (using NLTK stopwords)

filtered_disaster_corpus = [word for word in disaster_corpus if word not in stp_wrds_]

x,y = map(list, zip(*get_top_n_words(filtered_disaster_corpus, ngrams=(1,1), n=15)))

sns.barplot(x=y, y=x)

plt.show()
# Top 15 bigrams when there is a disaster tweet

filtered_disaster_corpus = [word for word in disaster_corpus if word not in stp_wrds_]

x,y = map(list, zip(*get_top_n_words(filtered_disaster_corpus, ngrams=(2,2), n=15)))

sns.barplot(x=y, y=x)

plt.show()
# Top 15 unigrams for non-disaster tweet

# removing stopwords

filtered_non_disaster_corpus = [word for word in non_disaster_corpus if word not in stp_wrds_]

x,y = map(list, zip(*get_top_n_words(filtered_non_disaster_corpus, ngrams=(1,1), n=15)))

sns.barplot(x=y, y=x)

plt.show()


# Top 15 bigrams when there is a non-disaster tweet

filtered_non_disaster_corpus = [word for word in non_disaster_corpus if word not in stp_wrds_]

x,y = map(list, zip(*get_top_n_words(filtered_non_disaster_corpus, ngrams=(2,2), n=15)))

sns.barplot(x=y, y=x)

plt.show()
# Top 15 unigrams in full train corpus

filtered_train_corpus = [word for word in train_corpus if word not in stp_wrds_]

x,y = map(list, zip(*get_top_n_words(filtered_train_corpus, n=15)))

sns.barplot(x=y, y=x)

plt.show()
# Top 15 bigrams in full train corpus

filtered_train_corpus = [word for word in train_corpus if word not in stp_wrds_]

x,y = map(list, zip(*get_top_n_words(filtered_train_corpus, ngrams=(2,2), n=15)))

sns.barplot(x=y, y=x)

plt.show()
# merge keyword, location and text columns

Train['Ttext'] = Train[['keyword','text']].apply(lambda x: ' '.join(x), axis = 1)

merged_Train = Train[['id','Ttext','target']]

merged_Train.head()
# merge keyword, location and text columns

Test['Ttext'] = Test[['keyword','text']].apply(lambda x: ' '.join(x), axis = 1)

merged_Test = Test[['id','Ttext']]

merged_Test.head()
# Function to check wheter the tokenised text still has the Stop Words or Punctuations

def Assess_Tokenize(word):

    return word not in stp_wrds_ and word not in list(string.punctuation) and len(word)>2



# Function to Clean the Text, by removing spaces, special characters, tokenizing, and Lemmatizing

def preprocess_text(text):

    

    cleared_text = []

    cleared_text_2 = []

    text = re.sub("'","",text)

    text = re.sub("(\\d|\\W)+"," ",text)

    text = re.sub(r"http\S+"," ",text)

    text = text.replace("nbsp", "")

    cleared_text = [word for word in word_tokenize(text.lower()) if Assess_Tokenize(word)]

    cleared_text_2 = [word for word in cleared_text if Assess_Tokenize(word)]

    return " ".join(cleared_text_2)



# Cleaing the whole Text Column    

merged_Train['Ttext'] = merged_Train['Ttext'].apply(preprocess_text)

merged_Test['Ttext'] = merged_Test['Ttext'].apply(preprocess_text)





merged_Train.head(20)
merged_Test.head(20)
Train_X = merged_Train.Ttext

Train_Y = merged_Train.target

Test_X = merged_Test.Ttext
# Vectorizing the text data

# Initiating TF-IDF ( Term Frequency - Inverse Document Frequency) 

TF_IDF = TfidfVectorizer()



# Fitting and transforming the Text to Feature Vectors

TrainX = TF_IDF.fit_transform((Train_X))

TestX = TF_IDF.transform((Test_X))

TrainX
from sklearn.model_selection import GridSearchCV

def gridSearch(X, y, model, params):

    cv = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', iid=False, cv=5, verbose = 5,refit='Accuracy')

    cv.fit(X, y)

    

    return cv.best_score_, cv.best_estimator_
def ForecastR(TrainX, TrainY, TestX):

    model = RandomForestClassifier(max_features='auto',n_estimators=1000)

    # Fit on training data

    model.fit(TrainX, TrainY)

    rf_predictions = model.predict(TestX)

    

    return rf_predictions
from xgboost import XGBClassifier

def ForecastG(TrainX, TrainY, TestX):

    model = XGBClassifier(n_estimators=1000)

    # Fit on training data

    model.fit(TrainX, TrainY)

    rf_predictions = model.predict(TestX)

    

    return rf_predictions
# predicting using Random Forest

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier



"""

params = {

    'n_estimators': [100,200,500,750,1000],

    'max_depth': [None,3,5,7,9],

    'min_samples_split' : [n for n in range(2, 11)], 

    'min_samples_leaf' : [n for n in range(1, 5)],

    'max_features': [None,'auto', 'log2']}



score, newModel = gridSearch(TrainX, Train_Y, model, params)



newModel

"""
TrY = ForecastR(TrainX, Train_Y, TrainX)

print('MSE:', mean_squared_error(Train_Y, TrY))

print('Accuracy:', accuracy_score(Train_Y, TrY))
pY = ForecastR(TrainX, Train_Y, TestX)
# Exporting the output

output = pd.DataFrame()

output['id'] = Test.id

output['target'] = pY

output.to_csv('submission7.csv', index = False)