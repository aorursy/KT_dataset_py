# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import spacy

import pandas as pd
nlp = spacy.load("en_core_web_lg")
reviews = pd.read_csv('../input/amazon-music-reviews/Musical_instruments_reviews.csv')

reviews.head()
def get_score(num):

    if num >= 4.0:

        return 2

    elif num==3.0:

        return 1

    else:

        return 0
new_column = pd.DataFrame({'label': [get_score(review.overall) for idx, review in reviews.iterrows()]})

reviews = reviews.merge(new_column, left_index = True, right_index = True)
reviews.head()
reviews.reviewText = reviews.reviewText.astype(str)
doc_vectors = []

with nlp.disable_pipes():

    for idx, review in reviews.iterrows():

        doc_vectors.append(nlp(review.reviewText).vector)

    

vectors = np.array([doc_vectors])

print(vectors.shape)
y = np.array(reviews.label.values.tolist())

print(y.shape)
vectors = vectors.reshape(vectors.shape[1:])
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(vectors, y, test_size=0.1, random_state=1)

from sklearn.svm import LinearSVC

svc = LinearSVC(random_state=1, dual=False, multi_class="ovr", max_iter=10000)
svc.fit(X_train, y_train)

print(f"Accuracy: {svc.score(X_test, y_test) * 100:.3f}%", )
def predictions(prediction):

    if prediction == 2:

        return "Positive"

    elif prediction == 1:

        return "Neutral"

    else:

        return "Negative"