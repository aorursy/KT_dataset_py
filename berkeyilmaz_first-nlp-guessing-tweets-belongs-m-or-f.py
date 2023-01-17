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
data = pd.read_csv("/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv", encoding = "latin1")



data = pd.concat([data.gender, data.description], axis=1)



data.dropna(axis = 0,inplace= True)

#women 1 men 0



data.gender = [1 if each == "female" else 0 for each in data.gender ]

data.gender
import re

first_description = data.description[4]

description = re.sub("[^a-zA-Z]", " ", first_description)# find except alphabet change " "



description = description.lower() # change all letters lower letter
#stopwords (irrelavent words) not important words



import nltk # natural language tool kit

nltk.download("stopwords") #i imported corfus file location.

from nltk.corpus import stopwords

nltk.download('punkt')

#from nltk.tokenize import word_tokenize

#world_tokenize is better than split method because when we use split we dont recognise should'nt or should.

description = nltk.word_tokenize(description)

#we use list comp.

description = [ word for word in description if not word in set(stopwords.words("english"))]

nltk.download('wordnet')

import nltk as nlp



lemma = nlp.WordNetLemmatizer()

description = [ lemma.lemmatize(word) for word in description]

# find root words for example thanks => thank memorise => memory

description = " ".join(description)

description_list = []



for description in data.description:

    description = re.sub("[^a-zA-Z]", " ", description)# find except alphabet change " "

    description = description.lower()

    description = nltk.word_tokenize(description)

    description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list.append(description)

    
from sklearn.feature_extraction.text import CountVectorizer #because of create bag words.

max_features = 5000 # we tried 500 =0.41 10k = 0.45, we find best acc this value number.



count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")



sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()



print("most comman use {} words: {}".format(max_features,count_vectorizer.get_feature_names()))

y = data.iloc[:,0].values

x= sparce_matrix

#train/test split



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.1, random_state=42)



from sklearn.naive_bayes import GaussianNB

nb= GaussianNB()

nb.fit(x_train,y_train)



y_pred = nb.predict(x_test).reshape(-1,1) # you can usereshape method in here or under code in acc cell as y_pred.reshape(-1,1) two method is okay.



print("accuracy: ", nb.score(y_pred,y_test))


