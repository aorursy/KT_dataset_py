# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_excel('/kaggle/input/predict-news-category/Data_Train.xlsx')
test = pd.read_excel('/kaggle/input/predict-news-category/Data_Test.xlsx')
submission = pd.read_excel('/kaggle/input/predict-news-category/Sample_submission.xlsx')

# Importing the Libraries

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

# Download the Following Modules once

nltk.download('stopwords')
nltk.download('wordnet')
print(train.shape)
train.head()
print(test.shape)
test.head()
#Printing the group by description of each category

train.groupby('SECTION').describe()
# Removing Duplicates to avoid Overfitting
train.drop_duplicates(inplace=True)

#A punctuations string for reference (added other valid characters from the dataset)

all_punctuations = string.punctuation + '‘’,:”][],'

#Method to remove punctuation marks from the data

def punc_remove(raw_text):
    no_punc = "".join([punc for punc in raw_text if punc not in all_punctuations])
    return no_punc

def stopword_remover(raw_text):
    words = raw_text.split()
    raw_text = " ".join([i for i in words if i not in stopwords.words('english')])
    return raw_text

lemmer = nltk.stem.WordNetLemmatizer()

def lem(words):
    return " ".join([lemmer.lemmatize(word,'v') for word in words.split()])


# All together 

def text_cleaner(raw):
    cleaned_text = stopword_remover(punc_remove(raw))
    return lem(cleaned_text)

#Applying the cleaner method to the entire data

train['CLEAN_STORY'] = train['STORY'].apply(text_cleaner)
from sklearn.feature_extraction.text import CountVectorizer


# Creating a bag of words Dictionery of words from the Data

bow_dictionery = CountVectorizer().fit(train['CLEAN_STORY'])

len(bow_dictionery.vocabulary_)

bow = bow_dictionery.transform(train['CLEAN_STORY'])

print(bow.shape)


from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer().fit(bow)

storytfidf = tfidf.transform(bow)


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(storytfidf, train['SECTION'])
#cleaning the test data

test['CLEAN_STORY'] = test['STORY'].apply(text_cleaner)
#Importing the Pipeline module from sklearn
from sklearn.pipeline import Pipeline

#Initializing the pipeline with necessary transformations and the required classifier
pipe = Pipeline([('Bow', CountVectorizer()),
                ('TfIdf', TfidfTransformer()),
                ('Classifier',MultinomialNB())])


#Fitting the training data to the pipeline
pipe.fit(train['CLEAN_STORY'],train['SECTION'])

#Predicting the SECTION 
test_pred = pipe.predict(test['CLEAN_STORY'])

#Writing the predictions to an excel sheet
pd.DataFrame(test_pred, columns = ['SECTION']).to_excel('predictions.xlsx')

print(test['CLEAN_STORY'],test_pred)
