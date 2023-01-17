import time

start_time = time.time()
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



from tqdm import tqdm

import numpy as np

import pandas as pd

import re



%matplotlib inline

import matplotlib.pyplot as plt
!pip install tensorflow-text==2.0.0 --user
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_text as text
#it helps to print full tweet , not a part

pd.set_option('display.max_colwidth', -1)

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
def clean(text):

    text = re.sub(r"http\S+", " ", text) # remove urls

    text = re.sub(r"RT ", " ", text) # remove RT

    # remove all characters if not in the list [a-zA-Z#@\d\s]

    text = re.sub(r"[^a-zA-Z#@\d\s]", " ", text)

    text = re.sub(r"[0-9]", " ", text) # remove numbers

    text = re.sub(r"\s+", " ", text) # remove extra spaces

    text = text.strip() # remove spaces at the beginning and at the end of string

    return text
train.text = train.text.apply(clean)

test.text = test.text.apply(clean)
train['text'][50:70]
test['text'][:5]
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
train_arrays, test_arrays, train_labels, test_labels = train_test_split(X_train,

                                                                        y_train,

                                                                        random_state =42,

                                                                        test_size=0.05)
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def xgboost_param_selection(X, y, nfolds):

    depth_m=[2,3,5,7,9]

    base_learners=[2,5,50,70,100]

    parameters=dict(n_estimators=base_learners , max_depth=depth_m)

    clf=RandomizedSearchCV(XGBClassifier(n_jobs=-1, class_weight='balanced') ,parameters, scoring='roc_auc', refit=True, cv=3)



    clf.fit(X, y)

#     cv_error=clf.cv_results_['mean_test_score']

#     train_error=clf.cv_results_['mean_train_score']

#     pred=clf.predict(X_train_bow)

#     score=roc_auc_score(y_train, pred)

#     estimator=clf.best_params_['n_estimators']

    clf.best_params_

#     depth=clf.best_params_['max_depth']

    return clf



# model = xgboost_param_selection(train_arrays,train_labels, 5)
def svc_param_selection(X, y, nfolds):

    Cs = [1.07]

    gammas = [2.075]

    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=8)

    grid_search.fit(X, y)

    grid_search.best_params_

    return grid_search



# model = svc_param_selection(train_arrays,train_labels, 5)
from keras.utils import np_utils 

from keras.datasets import mnist 

import seaborn as sns

from keras.initializers import RandomNormal

import time

# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4

# https://stackoverflow.com/a/14434334

# this function is used to update the plots for each epoch and error

def plt_dynamic(x, vy, ty, ax, colors=['b']):

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()

    

train_labels = np_utils.to_categorical(train_labels, 2) 

# y_test = np_utils.to_categorical(y_test, 10)



from keras.models import Sequential 

from keras.layers import Dense, Activation

output_dim = 2

input_dim = train_arrays.shape[1]



batch_size = 128 

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(input_dim,)))

model.add(Dense(128, activation='relu'))

# The model needs to know what input shape it should expect. 

# For this reason, the first layer in a Sequential model 

# (and only the first, because following layers can do automatic shape inference)

# needs to receive information about its input shape. 

# you can use input_shape and input_dim to pass the shape of input



# output_dim represent the number of nodes need in that layer

# here we have 10 nodes



model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))

model.summary()
train_labels.shape
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

nb_epoch = 10

history = model.fit(train_arrays,train_labels, batch_size=batch_size, epochs=nb_epoch, verbose=1)
# model.best_params_
pred = model.predict(test_arrays)
# cm = confusion_matrix(train_labels,pred.round())

# cm
# accuracy = accuracy_score(test_labels,pred)

# accuracy
test_pred = model.predict(X_test)

submission['target'] = test_pred.round().astype(int)

#submission.to_csv('submission.csv', index=False)
train_df_copy = train

train_df_copy = train_df_copy.fillna('None')

ag = train_df_copy.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'Count', 'target':'Disaster Probability'})



ag.sort_values('Disaster Probability', ascending=False).head(20)
count = 2

prob_disaster = 0.9

keyword_list_disaster = list(ag[(ag['Count']>count) & (ag['Disaster Probability']>=prob_disaster)].index)

#we print the list of keywords which will be used for prediction correction 

keyword_list_disaster
ids_disaster = test['id'][test.keyword.isin(keyword_list_disaster)].values

submission['target'][submission['id'].isin(ids_disaster)] = 1
submission.to_csv("submission.csv", index=False)

submission.head(10)
print("--- %s seconds ---" % (time.time() - start_time))