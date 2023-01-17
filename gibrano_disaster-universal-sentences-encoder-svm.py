!pip install --user tensorflow_text
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_text



from tqdm import tqdm

import numpy as np

import pandas as pd

import re
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
def clean(text):

    text = re.sub(r"http\S+", " ", text) # remove urls

    text = re.sub(r"RT ", " ", text) # remove rt

    text = re.sub(r"[^a-zA-Z\'\.\,\d\s]", " ", text) # remove special character except # @ . ,

    text = re.sub(r"[0-9]", " ", text) # remove number

    text = re.sub(r'\t', ' ', text) # remove tabs

    text = re.sub(r'\n', ' ', text) # remove line jump

    text = re.sub(r"\s+", " ", text) # remove extra white space

    text = text.strip()

    return text
train.text = train.text.apply(clean)

test.text = test.text.apply(clean)
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
X_train = []

for r in tqdm(train.text.values):

  emb = use(r)

  review_emb = tf.reshape(emb, [-1]).numpy()

  X_train.append(review_emb)



X_train = np.array(X_train)

y_train = train.target.values



X_test = []

for r in tqdm(test.text.values):

  emb = use(r)

  review_emb = tf.reshape(emb, [-1]).numpy()

  X_test.append(review_emb)



X_test = np.array(X_test)
train_arrays, test_arrays, train_labels, test_labels = train_test_split(X_train,y_train,test_size=0.05)
def svc_param_selection(X, y, nfolds):

    #Cs = [0.001, 0.01, 0.1, 1, 10]

    Cs = [1.070, 1.074, 1.075, 1.1, 1.125]

    #gammas = [0.001, 0.01, 0.1, 1]

    gammas = [2.065,2.075, 2.08]

    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=8)

    grid_search.fit(X, y)

    grid_search.best_params_

    return grid_search



model = svc_param_selection(train_arrays,train_labels, 5)
model.best_params_
pred = model.predict(test_arrays)
cm = confusion_matrix(test_labels,pred)

cm
accuracy = accuracy_score(test_labels,pred)

accuracy
test_pred = model.predict(X_test)

submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)