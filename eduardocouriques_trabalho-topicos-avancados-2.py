import pandas as pd

import numpy as np

dataset = pd.read_csv("/kaggle/input/tripadvisordataset/tripadvisor.csv", header=None)

# preparacao dos dados

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



# tokenizando as sentencas

dataset[1] = [word_tokenize(word) for word in dataset[1]]

print(dataset)
# remocao de stopwords

dataset[1] = dataset[1].apply(lambda x: [item for item in x if item not in stopwords.words("english")])
print(dataset.head())
dataset[1] = [' '.join(word) for word in dataset[1]]
print(dataset.head())
from sklearn import model_selection



X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset[1], 

                                                                    dataset[0], 

                                                                    test_size=0.3)
from sklearn.preprocessing import LabelEncoder



Encoder = LabelEncoder()

y_train = Encoder.fit_transform(y_train)

y_test = Encoder.fit_transform(y_test)
from sklearn.feature_extraction.text import TfidfVectorizer



Tfidf_vect = TfidfVectorizer(max_features=5000)

Tfidf_vect.fit(dataset[1])

Train_X_Tfidf = Tfidf_vect.transform(X_train)

Test_X_Tfidf = Tfidf_vect.transform(X_test)
Tfidf_vect.vocabulary_
#naive bayes

from sklearn import naive_bayes

from sklearn.metrics import accuracy_score



Naive = naive_bayes.MultinomialNB()

Naive.fit(Train_X_Tfidf, y_train)



predictions = Naive.predict(Test_X_Tfidf)

print("Acuracia NB:", accuracy_score(predictions, y_test)*100)