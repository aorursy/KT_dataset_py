# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Usual imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
import os
print(os.listdir("../input"))

# Plotly based imports for visualization
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
train_df.head()
# SpaCy Parser for questions
punctuations = string.punctuation
stopwords = list(STOP_WORDS)

parser = English()

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens
tqdm.pandas()
questions = train_df["question_text"].progress_apply(spacy_tokenizer)
tqdm.pandas()
test = test_df["question_text"].progress_apply(spacy_tokenizer)
# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

questions_vectorized = vectorizer.fit_transform(questions)
test_vectorized = vectorizer.transform(test)
from sklearn.naive_bayes import MultinomialNB as MNB
mnb = MNB()
mnb.fit(questions_vectorized,train_df["target"].values)
preds = mnb.predict(test_vectorized)
submission = pd.DataFrame.from_dict({'qid': test_df['qid']})
submission['prediction'] = preds
submission.to_csv('submission.csv', index=False)