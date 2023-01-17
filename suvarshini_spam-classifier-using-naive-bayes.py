import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Read data
df = pd.read_csv(
    '/kaggle/input/sms-spam-collection-dataset/spam.csv',
    encoding='latin-1',
    usecols=['v1', 'v2']).rename(
        columns={'v1':'target', 'v2':'content'})
df.head()
# Check null values
df.isna().sum()
# Check total spam and ham
df.target.value_counts()
# Check content length
df['len_content'] = df['content'].apply(lambda x:len(x))
df.head()
df.groupby('target')['len_content'].mean()
plt.figure(figsize=(10, 5))
df[df.target=='spam']['len_content'].plot(kind='hist', color='r', label='Spam', alpha=0.6)
df[df.target=='ham']['len_content'].plot(kind='hist', bins=35, color='b', label='Ham', alpha=0.5)
plt.legend(loc='upper right')
plt.xlabel('Content length')
plt.show()
steps = [
    ('tfidf', TfidfVectorizer(min_df=4)),
    ('model', MultinomialNB())]
pipeline = Pipeline(steps)

# Clean content
stopword = stopwords.words('english') + [
    'u', 'ü', 'ur', '4', '2', 'im', 'dont',
    'doin', 'ure', 'n', 'e', 'c', 'r', 'v',
    'k', 'y', 'x']

def before_text_clean(dataframe):
    # Train test split
    y = dataframe['target']
    X = dataframe['content']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model fit
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cf_matrix = confusion_matrix(y_test, y_pred)
    print('Accuracy: {}\nConfusion matrix:\n{}'.format(accuracy, cf_matrix))

    
def cleaning_text(text):
    # Remove punctuations
    text  = "".join([char for char in text if char not in string.punctuation])
        
    # Tokenize
    text = word_tokenize(text.lower())
    
    # Remove stopwords
    return " ".join([char for char in text if char not in stopword])                  
    

def after_text_clean(dataframe):
    # Text clean
    dataframe['clean_text'] = dataframe['content'].apply(lambda x:cleaning_text(x))
    
    # Train test split
    y = dataframe['target']
    X = dataframe['clean_text']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model fit
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cf_matrix = confusion_matrix(y_test, y_pred)
    print('Accuracy: {}\nConfusion matrix:\n{}'.format(accuracy, cf_matrix))
before_text_clean(df)
after_text_clean(df)
