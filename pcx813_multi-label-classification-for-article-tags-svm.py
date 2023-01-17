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
import json

import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer



infos_file = '/kaggle/input/linkedinfo/infos_0_3353_fulltext.json'

with open(infos_file, 'r') as f:

    infos = json.load(f)

    

content_length_threshold = 100



data_lst = []

tags_lst = []

for info in infos['content']:

    if len(info['fulltext']) < content_length_threshold:

        continue

    if len(info['description']) < content_length_threshold:

        continue

    data_lst.append({'title': info['title'],

                     'description': info['description'],

                     'fulltext': info['fulltext']})

    tags_lst.append([tag['tagID'] for tag in info['tags']])



df_data = pd.DataFrame(data_lst)

df_tags = pd.DataFrame(tags_lst)



# fit and transform the binarizer

mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(tags_lst)

Y.shape
from dataclasses import dataclass





@dataclass

class Dataset:

    data: pd.DataFrame

    target: pd.DataFrame

    target_names: pd.DataFrame

    target_decoded: pd.DataFrame



ds = Dataset(df_data, Y, mlb.classes_, tags_lst)

ds.data.head()
ds.target[:5]
ds.target_names[:5]
ds.target_decoded[:5]
from sklearn.feature_extraction.text import TfidfVectorizer



# Use the default parameters for now, use_idf=True in default

vectorizer = TfidfVectorizer()

# Use the short descriptions for now for faster processing

X = vectorizer.fit_transform(df_data.description)

X.shape

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC



# Use default parameters, and train and test with small set of samples.

clf = OneVsRestClassifier(LinearSVC())



from sklearn.utils import resample



X_sample, Y_sample = resample(X, Y, n_samples=1000, random_state=7)

clf.fit(X_sample, Y_sample)



X_sample_test, Y_sample_test = resample(X, Y, n_samples=10, random_state=1)

Y_sample_pred = clf.predict(X_sample_test)



# Inverse transform the vectors back to tags

pred_transformed = mlb.inverse_transform(Y_sample_pred)

test_transformed = mlb.inverse_transform(Y_sample_test)



for (t, p) in zip(test_transformed, pred_transformed):

    print(f'tags: {t} predicted as: {p}')
