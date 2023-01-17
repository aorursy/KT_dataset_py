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
from sklearn.feature_extraction.text import TfidfVectorizer
docA = "The car is driven on the road"

docB = "The truck is driven on the highway"
tfidf = TfidfVectorizer()
response = tfidf.fit_transform([docA, docB])
feature_names = tfidf.get_feature_names()

print(feature_names)
print(response)

print(response[1])
for col in response.nonzero()[1]:

    print (feature_names[col], ' - ', response[0, col])