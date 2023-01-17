# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv(r"../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding='latin-1')
data.head()
data.info()
data = pd.concat([data.gender,data.description],axis=1)
data.head()
data.dropna(axis = 0, inplace = True)
data.gender = [1 if each == "female" else 0 for each in data.gender]
data.gender.unique()
import re
first_description = data.description[4]

first_description
description = re.sub("[^a-zA-Z]"," ",first_description) # Don't choose a to z and A to Z, another ones replace with space
description
description = description.lower()

description
import nltk # natural language tool kit

nltk.download("stopwords") # downloading into corpus file

from nltk.corpus import stopwords # importing from corpus file
# description = description.split()

# tokenizer from nltk can be used instead of split

# but if we use split, words like "shouldn't" don't seperate like "should" and "not"

description = nltk.word_tokenize(description)
print(description)
description = [ word for word in description if not word in set(stopwords.words("english"))]
print(description)
import nltk as nlp
lemma = nlp.WordNetLemmatizer()

description = [ lemma.lemmatize(word) for word in description ]
print(description)
description = " ".join(description)

print(description)
description_list = []

for description in data.description:

    description = re.sub("[^a-zA-Z]"," ",description)

    description = description.lower()

    description = nltk.word_tokenize(description)

    description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description ]

    description = " ".join(description)

    description_list.append(description)
from sklearn.feature_extraction.text import CountVectorizer

max_features = 500
count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english",)
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print("Most common {} words : {}".format(max_features,count_vectorizer.get_feature_names()))
y = data.iloc[:,0].values # male ofr female classes

x = sparce_matrix
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)



print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))
