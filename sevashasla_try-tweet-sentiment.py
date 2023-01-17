import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

import scipy.spatial.distance as spdist

from sklearn.pipeline import Pipeline
fresh_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

fresh_data_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

fresh_data.head(10)
train_data = fresh_data.dropna()

test_data = fresh_data_test.dropna()



del fresh_data, fresh_data_test
vect = TfidfVectorizer()

vect.fit(list(pd.DataFrame(train_data.text).append(pd.DataFrame(test_data.text), ignore_index=True).text))

transformer_smaller = PCA(n_components=100)

transformer_smaller.fit(vect.transform(list(pd.DataFrame(train_data.text).append(pd.DataFrame(test_data.text), ignore_index=True).text)[:100]).toarray())
def my_transform(x):

    try:

        return transformer_smaller.transform(vect.transform([x]).toarray())

    except:

        return transformer_smaller.transform(vect.transform(x).toarray())
model = LogisticRegression() #хех, обучил

model.fit(my_transform(train_data.selected_text[:10000]), train_data.sentiment.values[:10000])
def my_predict(x):

    words = x.text.split()

    type_of = x.sentiment

    min_r = np.inf

    best_word = x.text

    for i in words:

        if (spdist.cosine(my_transform(i)[0], my_transform(words)[0]) < min_r) and (model.predict(my_transform(i)) == type_of):

            best_word = i

            min_r = spdist.euclidean(my_transform(i)[0], my_transform(words)[0])

    return best_word

    
res = pd.DataFrame()

id_arr = list(test_data.textID)

answ = []

for i in tqdm(range(len(id_arr))):

    answ.append(my_predict(test_data.iloc[i]))

res['textID'] = id_arr

res['selected_text'] = answ
res.to_csv('submission.csv', index=False)