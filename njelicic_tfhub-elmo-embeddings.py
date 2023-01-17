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
import tensorflow_hub as hub 

import tensorflow as tf





elmo = tf.saved_model.load('/kaggle/input/elmo-tf-hub/') 

texts  = ['the cat lies on the couch','the couch lies on the cat']

x = tf.constant(texts)

embeddings  = elmo.signatures["default"](x)['default']
embeddings.numpy()
for idx, value in enumerate(texts):

    print(value,embeddings.numpy()[idx])



    

print('are both embeddings equal?',embeddings.numpy()[0]==embeddings.numpy()[1])
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer().fit_transform(texts)

for idx, value in enumerate(texts):

    print(value,tfidf.todense()[idx])
    

print('are both embeddings equal?',tfidf.todense()[0]==tfidf.todense()[1])