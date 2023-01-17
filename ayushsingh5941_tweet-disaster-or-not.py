import pandas as pd

import numpy as np

import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import names, stopwords

import unicodedata

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
tweet_train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

tweet_test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

Id = tweet_test_df['id']
tweet_train_df.head()
tweet_train_df.isna().sum()
tweet_train_df.info()
tweet_train_df.shape
def data_imputation(data):

    data['keyword'].fillna(' ', inplace=True)

    data['location'].fillna(' ', inplace=True)

    return data
tweet_train_df = data_imputation(tweet_train_df)
tweet_test_df = data_imputation(tweet_test_df)
tweet_train_df.isna().sum()
tweet_train_df['text'] = tweet_train_df['text'] +' '+ tweet_train_df['location'] +' '+ tweet_train_df['keyword']

tweet_test_df['text'] = tweet_test_df['text'] +' '+ tweet_test_df['location'] +' '+ tweet_test_df['keyword']
target = tweet_train_df['target']
tweet_train_df.drop(['target', 'location', 'keyword', 'id'], axis=1, inplace=True)

tweet_test_df.drop(['location', 'keyword', 'id'], axis=1, inplace=True)
lemmetizer = WordNetLemmatizer()
all_names = set(names.words())
stop_words = set(stopwords.words('english'))
tf_idf = TfidfVectorizer(min_df=0.1, max_df=0.7)
def cleaned_string(string):

    # Removing all the digits

    string = re.sub(r'\d', '', string)

    

    # Removing accented data

    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    

    # Removing Mentions

    string = re.sub(r'@\w+', ' ', string)

    

    # Removing links 

    string = re.sub(r'(https?:\/\/)?([\da-zA-Z\.-\/\#\:]+)\.([\da-zA-Z\.\/\:\#]{0,9})([\/\w \.-\/\:\#]*)', ' ', string)

    

    # Removing all the digits special caharacters

    string = re.sub(r'\W', ' ', string)

        

    

    # Removing double whitespaces

    string = re.sub(r'\s+', ' ', string, flags=re.I)

    



    

    string = string.strip()

    

    #Removing all Single characters

    string = re.sub(r'\^[a-zA-Z]\s+','' , string)

    

    

    # Lemmetizing the string and removing stop words

    string = string.split()

    string = [lemmetizer.lemmatize(word) for word in string if word not in stop_words and word not in all_names]

    string = ' '.join(string)

    

    # Lowercasing all data

    string = string.lower()

        

    return string
def clean_text(data):

    for i in range(data.shape[0]):

        for j in range(data.shape[1]):

            data.iloc[i, j] = cleaned_string(data.iloc[i, j])

    return data

            

            

    
tweet_cleaned_test_df = clean_text(tweet_test_df)
tweet_cleaned_test_df.shape
tweet_cleaned_test_df.head()
tweet_cleaned_train_df = clean_text(tweet_train_df)
tweet_train_df.shape
tweet_cleaned_train_df.head()
X_train, X_valid, y_train, y_valid = train_test_split(tweet_cleaned_train_df['text'], target,random_state = 0)

catboost = LogisticRegression()
pipeline_sgd = Pipeline([

    ('tfidf',  TfidfVectorizer()),

    ('nb', catboost,)

])
model = pipeline_sgd.fit(X_train, y_train)
y_predict = model.predict(X_valid)
print(classification_report(y_valid, y_predict))
y_pred_test = model.predict(tweet_cleaned_test_df['text'])
# Saving result on test set

output = pd.DataFrame({'Id': Id,

                       'target': y_pred_test})



output.to_csv(r'submission.csv', index=False)