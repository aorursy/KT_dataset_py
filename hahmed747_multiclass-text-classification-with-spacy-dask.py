import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



PATH = '/kaggle/input/amazon-fine-food-reviews/Reviews.csv'  
import numpy as np

import pandas as pd

import string

import spacy

import pickle



import dask.dataframe as dd

from dask.distributed import Client

from dask_ml.model_selection import train_test_split

from dask_ml.feature_extraction.text import HashingVectorizer

from dask_ml.wrappers import Incremental

from dask_ml.metrics import accuracy_score



from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.utils import class_weight

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report

from dask.distributed import Client



client = Client(threads_per_worker=2,

                n_workers=5, memory_limit='3GB')

client
reqd = ['Text', 'Score']

Reviews_df = dd.read_csv(PATH,

                         usecols = reqd,

                         blocksize=20e6,

                         dtype={'Score': 'float'},

                         engine='python',

                         encoding='utf-8',

                         error_bad_lines=False)
Reviews_df.info()
Reviews_df
# frac = 1.0

# Reviews_df = Reviews_df.sample(frac=frac, replace=True)
Reviews_df['Score'].value_counts().compute()
X = Reviews_df['Text']

ylabels = Reviews_df['Score'] 
keys = np.unique(ylabels.compute())

values = class_weight.compute_class_weight('balanced',

                                           keys,

                                           ylabels.compute())

class_weights = dict(zip(keys, values))
class_weights
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2)
# Create our list of punctuation marks

punctuations = string.punctuation



# Create our list of stopwords

nlp = spacy.load('en')

stop_words = spacy.lang.en.stop_words.STOP_WORDS



# Load English tokenizer, tagger, parser, NER and word vectors

parser = English()
def spacy_tokenizer(sentence):

    # Creating our token object, which is used to create documents with linguistic annotations.

    mytokens = parser(sentence)



    # Lemmatizing each token and converting each token into lowercase

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]



    # Removing stop words

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]



    # return preprocessed list of tokens

    return mytokens
from dask_ml.feature_extraction.text import HashingVectorizer

hw_vector = HashingVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1, 2), n_features=2**20)
%time

Feature_pipeline = Pipeline([('vectorizer', hw_vector)])

Pipeline_Model = Feature_pipeline.fit(X_train.values)
Text_preprocess_pipe = pickle.dumps(Pipeline_Model)
Pipeline_Model = pickle.loads(Text_preprocess_pipe)
%time

X_transformed = Pipeline_Model.transform(X_train)
%%time

import joblib

estimator = SGDClassifier(random_state=10, max_iter=200, loss='modified_huber',class_weight = class_weights, n_jobs=-1)

classifier = Incremental(estimator)

Model = classifier.fit(X_transformed,

               y_train,

               classes=list(class_weights.keys()))
predictions = Model.predict(Pipeline_Model.transform(X_test))

predictions
accuracy_score(y_test, predictions)
ML_Model = pickle.dumps(Model)
%time

Model = pickle.loads(ML_Model)

# X = Model.predict_proba(X_transformed).compute()
%time

x_test_transformed = Pipeline_Model.transform(X_test)

y_pred = Model.predict(x_test_transformed).compute()
%time

print(classification_report(y_train,

                            Model.predict(Pipeline_Model.transform(X_train)).compute()))
%time

print(classification_report(y_test, y_pred))
client.close()