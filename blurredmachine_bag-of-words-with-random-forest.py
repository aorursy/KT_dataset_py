import pandas as pd     

import numpy as np

from bs4 import BeautifulSoup

import re

import nltk

# nltk.download()

from nltk.corpus import stopwords # Import the stop word list



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/labeledTrainData.tsv", 

                              header=0, 

                              delimiter="\t", 

                              quoting=3)



df_test = pd.read_csv("/kaggle/input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/testData.tsv",

                             header=0, 

                             delimiter="\t", 

                             quoting=3)
print(df_train.shape)

print(df_test.shape)
df_train.info()
print(df_train.columns.values)

print(df_test.columns.values)
df_train['review'][0]


bs_data = BeautifulSoup(df_train["review"][0])

print(bs_data.get_text())
letters_only = re.sub("[^a-zA-Z]", " ", bs_data.get_text() )

print(letters_only)
lower_case = letters_only.lower()  

words = lower_case.split()  

print(words)
print(stopwords.words("english") )
words = [w for w in words if not w in stopwords.words("english")]

print(words)
training_data_size = df_train["review"].size

testing_data_size = df_test["review"].size



print(training_data_size)

print(testing_data_size)
def clean_text_data(data_point, data_size):

    review_soup = BeautifulSoup(data_point)

    review_text = review_soup.get_text()

    review_letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    review_lower_case = review_letters_only.lower()  

    review_words = review_lower_case.split() 

    stop_words = stopwords.words("english")

    meaningful_words = [x for x in review_words if x not in stop_words]

    

    if( (i)%2000 == 0 ):

        print("Cleaned %d of %d data (%d %%)." % ( i, data_size, ((i)/data_size)*100))

        

    return( " ".join( meaningful_words)) 

    
# clean_train_data_list = []

# clean_test_data_list = []
df_train.head()
for i in range(training_data_size):

    df_train["review"][i] = clean_text_data(df_train["review"][i], training_data_size)

print("Cleaning training completed!")
for i in range(testing_data_size):

    df_test["review"][i] = clean_text_data(df_test["review"][i], testing_data_size)

print("Cleaning validation completed!")
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 5000) 
X_train, X_cv, Y_train, Y_cv = train_test_split(df_train["review"], df_train["sentiment"], test_size = 0.3, random_state=42)
X_train = vectorizer.fit_transform(X_train)

X_train = X_train.toarray()

print(X_train.shape)
X_cv = vectorizer.transform(X_cv)

X_cv = X_cv.toarray()

print(X_cv.shape)
X_test = vectorizer.transform(df_test["review"])

X_test = X_test.toarray()

print(X_test.shape)
vocab = vectorizer.get_feature_names()

print(vocab)
distribution = np.sum(X_train, axis=0)



for tag, count in zip(vocab, distribution):

    print(count, tag)
forest = RandomForestClassifier() 

forest = forest.fit( X_train, Y_train)
predictions = forest.predict(X_cv) 

print("Accuracy: ", accuracy_score(Y_cv, predictions))
result = forest.predict(X_test) 

output = pd.DataFrame( data={"id":df_test["id"], "sentiment":result} )

output.to_csv( "submission.csv", index=False, quoting=3 )
