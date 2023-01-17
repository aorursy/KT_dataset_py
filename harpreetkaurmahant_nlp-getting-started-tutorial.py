import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import time

import math

import keras



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import preprocessing





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import warnings; warnings.simplefilter('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.width',1000000)

pd.set_option('display.max_columns', 500)



score_df = pd.DataFrame(columns={'Model Description','Score'})

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.head(50)
print(train_df.info())
print(train_df.isnull().any())
print(test_df.isnull().any())
print(train_df.shape)
#printing records



print( train_df[train_df["target"] == 0]["text"].values[1] )

print( train_df[train_df["target"] == 1]["text"].values[1] )
import seaborn as sns

from matplotlib import pyplot as plt



fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

plt.tight_layout()



labels=['Disaster Tweet','No Disaster']

size=  [train_df['target'].mean()*100,abs(1-train_df['target'].mean())*100]

explode = (0, 0.1)

#ig1,ax1 = plt.subplots()

axes[0].pie(size,labels=labels,explode=explode,shadow=True,

            startangle=90,autopct='%1.1f%%')

sns.countplot(x=train_df['target'], hue=train_df['target'], ax=axes[1])

plt.show()
count_vectorizer = feature_extraction.text.CountVectorizer()



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
count_vectorizer
print(example_train_vectors)
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df["text"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text"])
train_vectors
test_vectors
## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head(100)
sample_submission.to_csv("submission.csv", index=False)