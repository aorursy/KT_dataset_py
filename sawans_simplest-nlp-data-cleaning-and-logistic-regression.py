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
import numpy as np

import pandas as pd

import nltk
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
#disaster tweets

train_df[train_df['target']==1][:5]['text']
#non-disaster tweets

train_df[train_df['target']==0][:5]['text']
nltk.download('stopwords')

from nltk.corpus import stopwords
X = train_df['text']

y = train_df['target']
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=0)

X_test = test_df['text']
# Importing word_tokenize to tokenize the text before processing

import re 

nltk.download('punkt')

from nltk.tokenize import word_tokenize
replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')

bad_symbols_re = re.compile('[^0-9a-z #+_]')

links_re = re.compile('(www|http)\S+')



Stopwords = set(stopwords.words('english'))

Stopwords.remove('no')

Stopwords.remove('not')



lemmatizer = nltk.stem.WordNetLemmatizer()
def text_prepare(text):

    """

        text: a string

        

        return: modified initial string

    """

    

    text = text.lower()  # lowercase text

    text = re.sub(replace_by_space_re," ",text) # replace symbols by space

    text = re.sub(bad_symbols_re, "",text) # remove bad symbols

    text = re.sub(links_re, "",text) # remove hyperlinks

    

    word_tokens = word_tokenize(text) # Creating word tokens out of the text

    

    filtered_tokens=[]

    for word in word_tokens:

        if word not in Stopwords:

            filtered_tokens.append(lemmatizer.lemmatize(word))

    

    text = " ".join(word for word in filtered_tokens)

    return text
X_train = [text_prepare(x) for x in X_train]

X_val = [text_prepare(x) for x in X_val]

X_test = [text_prepare(x) for x in X_test]
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vectorizer = TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2))

tfidf_vectorizer.fit(X_train)



X_train_tfidf = tfidf_vectorizer.transform(X_train)

X_val_tfidf = tfidf_vectorizer.transform(X_val)

X_test_tfidf = tfidf_vectorizer.transform(X_test)
X_train_tfidf.shape
from sklearn.linear_model import LogisticRegression



log_reg_classifier = LogisticRegression(penalty='l2',C=1,solver='liblinear')

log_reg_classifier.fit(X_train_tfidf,y_train)
y_val_predicted = log_reg_classifier.predict(X_val_tfidf)
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score
accuracy = accuracy_score(y_val,y_val_predicted)

f1Score =f1_score(y_val,y_val_predicted,average='weighted')



print('Accuracy is :',accuracy)

print('F1 score is :',f1Score)
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
y_test = log_reg_classifier.predict(X_test_tfidf)

sample_submission['target']= y_test

sample_submission.head()
sample_submission.to_csv("Submission_LogisticRegression_Final",index=False)