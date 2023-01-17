import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split # K-FOLD CV can be done but will take too much time.
from sklearn.cluster import KMeans # Kmeans classifier

# Removing numerical figures, symbols and stopwords
def clean_description(description, stmr):
    description = re.sub('[^a-zA-Z]',' ',description).lower()
    description_words = description.split(' ')
    description_words = [x for x in description_words if x not in stopwords.words('english')]
    description = ' '.join([stmr.stem(x) for x in description_words])
    return description

# Fitting the vectorizer with a training corpus
def fit_vectorizer(train_data, vectorizer):
    ps = PorterStemmer()
    text_corpus = " "
    for item in train_data['description']:
        text_corpus = text_corpus + clean_description(item, ps)
    text_corpus = [text_corpus]
    vectorizer.fit(text_corpus)
    return vectorizer

# Tranforming a text sequence to a vector
def transform_using_vectorizer(content, vectorizer):
    return vectorizer.transform([content]).toarray()[0]

# Creating and fitting a kmeans classifier based on training data
def create_and_fit_classfier(train_data):
    # The choice of n=4 is arbitarary. 
    clf = KMeans(n_clusters=4)
    clf.fit(train_data)
    print("classfier built")
    return clf

if __name__ == "__main__":
    df = pd.read_csv('../input/winemag-data_first150k.csv')
    df = df[:1000] # Trying with a sample of 1000 
    pd.DataFrame(df)
    size = len(df)

    # Splitting the dataset into training and testing
    # random_state of 42 results in the same split every time it's run this way we can reproduce the test
    df_train, df_test = train_test_split(df, test_size=int(size/2), train_size=int(size/2), random_state=42, shuffle=True)

    # Creating a TfIdf vectorizer and fitting it with the training data (only description field)
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
    vectorizer = fit_vectorizer(df_train, vectorizer)

    # Obtaining an array of transformed descriptions to be passed to the kmeans classifier
    descriptions = []
    for item in df_train['description']:
        descriptions.append(transform_using_vectorizer(item, vectorizer))

    kmeans_classifier = create_and_fit_classfier(descriptions)
    index_description_list = zip(df_test['Unnamed: 0'], df_test['description'], df_test['variety'], df_test['designation'])
    predictions = []
    for item in index_description_list:
        predictions.append(kmeans_classifier.predict([transform_using_vectorizer(item[1], vectorizer)])[0])
    print(pd.DataFrame({'id': df_test['Unnamed: 0'], 'variety': df_test['variety'], 'designation': df_test['designation'], 'grouping': predictions}))
    print("All wines that have the same GroupId must theoretically be similar, the predicition is solely based on the description given")
