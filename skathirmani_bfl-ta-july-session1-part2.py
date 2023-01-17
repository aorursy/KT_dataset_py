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
from gensim.parsing.preprocessing import remove_stopwords

from gensim.parsing.porter import PorterStemmer



stemmer = PorterStemmer()
test = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')

test.head()
train = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv').sample(100000)

train.head()
# docs will be a pandas series

def clean_documents(docs):

    stemmer = PorterStemmer()

    docs_clean = docs.str.lower()

    docs_clean = docs_clean.str.replace('[^a-z\s]', '')

    docs_clean = docs_clean.apply(lambda doc: remove_stopwords(doc))

    docs_clean = pd.Series(stemmer.stem_documents(docs_clean), index=docs.index)

    return docs_clean
train_cleaned = clean_documents(train['question_text'])

train_cleaned.head()
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, f1_score

train_x, validate_x, train_y, validate_y = train_test_split(train_cleaned,

                                                           train['target'],

                                                           test_size=0.2,

                                                           random_state=1)

train_x.shape, validate_x.shape, train_y.shape, validate_y.shape
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(min_df=2,stop_words='english',).fit(train_x)

vocab = vectorizer.get_feature_names()

train_dtm = vectorizer.transform(train_x)

validate_dtm = vectorizer.transform(validate_x)
#df_train_dtm = pd.DataFrame(train_dtm.toarray(), columns=vocab, index=train_x.index)

#df_validate_dtm = pd.DataFrame(validate_dtm.toarray(), columns=vocab, index=validate_x.index)
nb_model = MultinomialNB().fit(train_dtm, train_y)

pred_validate_y = pd.Series(nb_model.predict(validate_dtm), index=validate_y.index)
print(accuracy_score(validate_y, pred_validate_y))

print(f1_score(validate_y, pred_validate_y))
test_docs = clean_documents(test['question_text'])
test_dtm = vectorizer.transform(test_docs)

#df_test_dtm = pd.DataFrame(test_dtm.toarray(), index=test_docs.index, columns=vocab)

pred_test_y = pd.Series(nb_model.predict(test_dtm))
sample = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/sample_submission.csv')

sample.head()
submission = pd.DataFrame({

    'qid': test['qid'],

    'prediction': pred_test_y

})

submission.to_csv('submission.csv', index=False)