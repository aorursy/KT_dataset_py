# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from IPython.display import display

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

plt.style.use('ggplot')

WIDE = (12,5)

WIDER = (16,6)

plt.rcParams['figure.figsize'] = WIDER

%aimport svds.util 

%aimport svds.egv



import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

%matplotlib inline



from IPython.display import display, HTML ,display_html



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv', sep='\t')

df.head()
df.groupby('rating').count()
df.describe()
# Add new column which have length of verified reviews

df['length'] = df['verified_reviews'].apply(len)

df.head()
# Distribution of length of review

sns.kdeplot(df['length'], shade=True);
df.length.describe()
# Lets Clean the texts



corpus=[]

for i in range(0,3150):

    review = re.sub('[^a-zA-Z]', ' ', df['verified_reviews'][i] )

    review=review.lower()

    review=review.split()

    ps=PorterStemmer()

    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review=' '.join(review)

    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=1500)

X=cv.fit_transform(corpus).toarray()

y=df.iloc[:,4].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
import xgboost as xgb

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
cm
accuracy_score(y_test, y_pred)