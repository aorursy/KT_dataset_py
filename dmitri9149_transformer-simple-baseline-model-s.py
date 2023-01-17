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

import tensorflow_text as textb
#print full tweet , not a part

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 100)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

length_train = len(train.index)

length_train
# the code in the cell is taken from 

# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert

df_mislabeled = train.groupby(['text']).nunique().sort_values(by='target', ascending=False)

df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']

index_misl = df_mislabeled.index.tolist()



lenght = len(index_misl)



print(f"There are {lenght} equivalence classes with mislabelling")
index_misl
train_nu_target = train[train['text'].isin(index_misl)].sort_values(by = 'text')

train_nu_target.head(60)
num_records = train_nu_target.shape[0]

length = len(index_misl)

print(f"There are {num_records} records in train set which generate {lenght} equivalence classes with mislabelling (raw text, no cleaning)") 
copy = train_nu_target.copy()

classes = copy.groupby('text').agg({'keyword':np.size, 'target':np.mean}).rename(columns={'keyword':'Number of records in train set', 'target':'Target mean'})



classes.sort_values('Number of records in train set', ascending=False).head(20)
majority_df = train_nu_target.groupby(['text'])['target'].mean()

#majority_df.index
def relabel(r, majority_index):

    ind = ''

    if r['text'] in majority_index:

        ind = r['text']

#        print(ind)

        if majority_df[ind] <= 0.5:

            return 0

        else:

            return 1

    else: 

        return r['target'] 
train['target'] = train.apply( lambda row: relabel(row, majority_df.index), axis = 1)
new_df = train[train['text'].isin(majority_df.index)].sort_values(['target', 'text'], ascending = [False, True])

new_df.head(15)
# the code in the cell is taken from 

# https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert

df_mislabeled = train.groupby(['text']).nunique().sort_values(by='target', ascending=False)

df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']

index_misl = df_mislabeled.index.tolist()

#index_dupl[0:50]

len(index_misl)
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

                                                                        test_size=0.20)
def svc_param_selection(X, y, nfolds):

    Cs = [1.07]

    gammas = [2.075]

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