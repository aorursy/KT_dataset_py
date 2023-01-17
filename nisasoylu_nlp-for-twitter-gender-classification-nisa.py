# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
whole_dataset = pd.read_csv("/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv", encoding = "latin1")
dataset = pd.concat([whole_dataset.gender, whole_dataset.description], axis = 1)
dataset.head(20)
dataset = dataset.dropna()
dataset.head(20)
dataset.gender = [1 if person == "female" else 0 for person in dataset.gender]
dataset.head(10)
import re
first_description = dataset.description[4]

first_description
first_description = re.sub("[^a-zA-z]", " ",first_description)

first_description
first_description = first_description.lower()

first_description
import nltk # natural language took kit

nltk.download("stopwords")

from nltk.corpus import stopwords
first_description = nltk.word_tokenize(first_description)

first_description
first_description = [word for word in first_description if not word in set(stopwords.words("english"))]
first_description
import nltk as nlp



lemma = nlp.WordNetLemmatizer()

first_description = [lemma.lemmatize(i) for i in first_description]
first_description
first_description = " ".join(first_description)
first_description
description_list = []

for description in dataset.description:

    description = re.sub("[^a-zA-z]", " ",description)

    description = description.lower()

    description = nltk.word_tokenize(description)

    description = [word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [lemma.lemmatize(i) for i in description]

    description = " ".join(description)

    description_list.append(description)
from sklearn.feature_extraction.text import CountVectorizer



max_features = 5000

count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print("5000 most common words: ", count_vectorizer.get_feature_names())
x = sparce_matrix
x
y = dataset.iloc[:,0].values



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(x_train, y_train)
predictions = nb.predict(x_test)
print("Accuracy: ", nb.score(predictions.reshape(-1,1), y_test))