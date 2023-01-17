import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings('ignore', category=DeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'

%matplotlib inline
temp = pd.read_csv('../input/consumer-reviews-of-amazon-products/1429_1.csv')
temp.head()
temp.info()
# create a new dataframe consist of only text and rating
df = pd.DataFrame()
df[['text', 'rating']] = temp[['reviews.text', 'reviews.rating']]
df.head()
# Investigate how many rows of have a Null values
df.isnull().sum()
# drop the rows with Null values 
df.dropna(inplace=True)
df.info()
df['label'] = df['rating'].apply(lambda x : 1 if x >= 4 else 0) 

# drop the unneeded column of ratings
df.drop(labels=['rating'], axis=1, inplace=True)

df.head()
def remove_pattern(text, pattern):
    """
    Docstring: 
    
    remove any pattern from the input text.
    
    Parameters
    ----------
    text: string input, the text to clean.
    pattern : string input, the pattern to remove from the text input.
    
    Returns
    -------
    a cleaned string.
    
    """
    
    # find all the pattern in the input text and return a list of postion indeces 
    r = re.findall(pattern, text)
    
    # replace the pattern with an empty space
    for i in r: text = re.sub(pattern, '', text)
    
    return text
# lower case every word to ease the upcoming processes 
df['text'] = df['text'].str.lower()

# tokenize the text to search for any stop words to remove it
df['tokenized_text'] = df.text.apply(lambda x : x.split())

# creating a set of stopwords(if you wonder why set cuz it is faster than a list)
stopWords = set(nltk.corpus.stopwords.words('english'))
df['tokenized_text'] = df['tokenized_text'].apply(lambda x : [word for word in x if not word in stopWords])

# create a word net lemma
lemma = nltk.stem.WordNetLemmatizer()
pos = nltk.corpus.wordnet.VERB
df['tokenized_text'] = df['tokenized_text'].apply(lambda x : [lemma.lemmatize(word, pos) for word in x])

# remove any punctuation
df['tokenized_text'] = df['tokenized_text'].apply(lambda x : [ remove_pattern(word,'\.') for word in x])

# rejoin the text again to get a cleaned text
df['cleaned_text'] = df['tokenized_text'].apply(lambda x : ' '.join(x))

df.drop(labels=['tokenized_text'], axis=1, inplace=True)

df.head()
from sklearn.feature_extraction.text import CountVectorizer


# perform vectorization on our cleaned text 
bow_vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english', max_features=1000)

bow_features = bow_vectorizer.fit_transform(df['cleaned_text'])

bow_df = pd.DataFrame(bow_features.toarray(), columns=bow_vectorizer.get_feature_names())

bow_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_Vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')

tfidf_features = tfidf_Vectorizer.fit_transform(df['cleaned_text'])

tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_Vectorizer.get_feature_names())

tfidf_df.head()
from sklearn.model_selection import train_test_split

X_train_bow, X_metric_bow, y_train_bow, y_metric_bow = train_test_split(bow_df, df['label'], test_size=0.2, random_state=42)
X_test_bow, X_valid_bow, y_test_bow, y_valid_bow = train_test_split(X_metric_bow, y_metric_bow, test_size=0.5, random_state=42)


X_train_tfidf, X_metric_tfidf, y_train_tfidf, y_metric_tfidf = train_test_split(tfidf_df, df['label'], test_size=0.2, random_state=42)
X_test_tfidf, X_valid_tfidf, y_test_tfidf, y_valid_tfidf = train_test_split(X_metric_tfidf, y_metric_tfidf, test_size=0.5, random_state=42)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

clf_bow = clf_tfidf = AdaBoostClassifier(n_estimators=100, learning_rate=0.001)
clf_bow.fit(X_train_bow, y_train_bow)
clf_tfidf.fit(X_train_tfidf, y_train_tfidf)
pred_bow   = clf_bow.predict(X_test_bow)
pred_tfidf = clf_tfidf.predict(X_test_tfidf)
from colorama import Fore, Style

print(f'AdaBoost Classifier Results: \n',
      f'{Fore.RED}Bag of words{Style.RESET_ALL} \n',
      f'Accuracy Socre: {Fore.LIGHTBLUE_EX}%0.2f %%{Style.RESET_ALL} \n'%(100 * accuracy_score(y_test_bow, pred_bow)))
print(classification_report(y_test_bow, pred_bow))

print(f'{Fore.RED}TF-IDF{Style.RESET_ALL} \n',
      f'Accuracy Socre: {Fore.LIGHTBLUE_EX}%0.2f %%{Style.RESET_ALL} \n'%(100 * accuracy_score(y_test_tfidf, pred_tfidf)))
print(classification_report(y_test_tfidf, pred_tfidf))
from sklearn.neighbors import KNeighborsClassifier

clf_bow_knn = clf_tfidf_knn = KNeighborsClassifier()
clf_bow_knn.fit(X_train_bow, y_train_bow)
clf_tfidf_knn.fit(X_train_tfidf, y_train_tfidf)
pred_bow_knn   = clf_bow_knn.predict(X_test_bow)
pred_tfidf_knn = clf_tfidf_knn.predict(X_test_tfidf)
print(f'KNN Classifier Results: \n',
      f'{Fore.RED}Bag of words{Style.RESET_ALL} \n',
      f'Accuracy Socre: {Fore.LIGHTBLUE_EX}%0.2f %%{Style.RESET_ALL} \n'%(100 * accuracy_score(y_test_bow, pred_bow_knn)))
print(classification_report(y_test_bow, pred_bow_knn))

print(f'{Fore.RED}TF-IDF{Style.RESET_ALL} \n',
      f'Accuracy Socre: {Fore.LIGHTBLUE_EX}%0.2f %%{Style.RESET_ALL} \n'%(100 * accuracy_score(y_test_tfidf, pred_tfidf_knn)))
print(classification_report(y_test_tfidf, pred_tfidf_knn))
