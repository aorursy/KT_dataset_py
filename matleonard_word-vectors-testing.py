import numpy as np

import spacy



# Need to load the large model to get the vectors

nlp = spacy.load('en_core_web_lg')
# Disabling other pipes because we don't need them and it'll speed up this part a bit

text = "These vectors can be used as features for machine learning models."

with nlp.disable_pipes():

    vectors = np.array([token.vector for token in  nlp(text)])
vectors.shape
import pandas as pd



# Loading the spam data

# ham is the label for non-spam messages

spam = pd.read_csv('../input/nlp-course/spam.csv')
# This takes a minute or so

with nlp.disable_pipes():

    doc_vectors = np.array([nlp(text).vector for text in spam.text])
doc_vectors
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(doc_vectors, spam.label, test_size=0.1, random_state=1)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



model = LogisticRegression(random_state=1, solver='lbfgs')

model.fit(X_train, y_train)



accuracy = metrics.accuracy_score(y_test, model.predict(X_test))

print("Accuracy: ", accuracy)
from sklearn.svm import LinearSVC



# Set dual=False to speed up training, and it's not needed

svc = LinearSVC(random_state=1, dual=False, max_iter=10000)

svc.fit(X_train, y_train)

print("Accuracy: ", svc.score(X_test, y_test))