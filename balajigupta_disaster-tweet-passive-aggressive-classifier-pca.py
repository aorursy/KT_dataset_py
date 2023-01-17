import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix
import itertools
from sklearn.linear_model import PassiveAggressiveClassifier
df_X = pd.read_csv('../input/nlp-getting-started/train.csv')

labels = df_X.target

labels.head()

df_Xtrain, df_Xtest, df_Ytrain, df_Ytest = train_test_split(df_X['text'], labels, test_size=0.55950, random_state=5)





tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(df_Xtrain)
tfidf_test = tfidf_vectorizer.transform(df_Xtest)
pac = PassiveAggressiveClassifier (max_iter=120)
pac.fit(tfidf_train, df_Ytrain)
Y_pred = pac.predict(tfidf_test)
Score = accuracy_score(df_Ytest, Y_pred)
print(Score)

import pickle

saved_model = pickle.dumps(pac)

pac_from_pickle = pickle.loads(saved_model)
z = pd.read_csv('../input/nlp-getting-started/test.csv')

Z = z['text']

tfidf_real = tfidf_vectorizer.transform(Z)

tfidf_real.shape

final_result = pac_from_pickle.predict(tfidf_real)

sub_z = pd.DataFrame({"id" : z['id'], "target" : final_result})
sub_z.to_csv("sub_z.csv", index = None)