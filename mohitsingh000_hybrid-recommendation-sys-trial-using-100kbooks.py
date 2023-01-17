# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import csv

r =  csv.DictReader(open("/kaggle/input/float-ratings/UpdatedData.csv"), delimiter=",")

b =  csv.DictReader(open("/kaggle/input/book-recommendation-sysguide/books.csv"), delimiter=",")
import pandas as pd

from itertools import islice

import json

import csv
import json

from itertools import islice

for line in islice(r, 3 ):

    print(json.dumps(line, indent=4))
for line in islice(b, 1):

    print(json.dumps(line, indent=4))

rr= pd.read_csv("/kaggle/input/float-ratings/UpdatedData.csv")
NUM_THREADS = 9

NUM_COMPONENTS = 30

NUM_EPOCHS = 3

ITEM_ALPHA = 1e-6

from lightfm.data import Dataset



dataset = Dataset()

dataset.fit((rr['user_id']),

            (rr['book_id']),

             

          )
tags = rr['rating']
tags.head
num_users, num_items = dataset.interactions_shape()

print('Num users: {}, num_items {}.'.format(num_users, num_items))
dataset.fit_partial(items=(x['book_id'] for x in b),

                    item_features=(x['isbn'] for x in b))
(interactions, weights) = dataset.build_interactions([tuple(i) for i in rr.drop(['rating'], axis = 1).values])
interactions
item_features = dataset.build_item_features(((x['book_id'], [x['isbn']])

                                              for x in b))

print(repr(item_features))
from lightfm import LightFM

model = LightFM(loss='warp',

                item_alpha=ITEM_ALPHA,

                no_components=NUM_COMPONENTS)
model = model.fit(interactions,

                item_features=item_features,

                epochs=NUM_EPOCHS,

                num_threads=NUM_THREADS)
from lightfm.evaluation import auc_score



train_auc = auc_score(model,

                      interactions,

                      item_features=item_features,

                      num_threads=NUM_THREADS).mean()

print('Hybrid training set AUC: %s' % train_auc)
# def get_similar_tags(model, tag_id):

#     # Define similarity as the cosine of the angle

#     # between the tag latent vectors



#     # Normalize the vectors to unit length

#     tag_embeddings = (model.item_embeddings.T

#                       / np.linalg.norm(modinel.item_embeddings, axis=1)).T



#     query_embedding = tag_embeddings[tag_id]

#     similarity = np.dot(tag_embeddings, query_embedding)

#     most_similar = np.argsort(-similarity)[1:4]



#     return most_similar





# for tag in (u'bayesian', u'regression', u'survival'):

#     tag_id = tag_labels.tolist().index(tag)

#     print('Most similar tags for %s: %s' % (tag_labels[tag_id],

#                                             tag_labels[get_similar_tags(model, tag_id)]))
def sample_recommendation(model, data, user_ids):





    n_users, n_items = interactions.shape



    for user_id in user_ids:

        known_positives = bg['original_title'][interactions.tocsr()[user_id].indices]



        scores = model.predict(user_id, np.arange(n_items))

        top_items = bg['original_title'][np.argsort(-scores)]



        print("User %s" % user_id)

        print("     Known positives:")



        for x in known_positives[:3]:

            print("        %s" % x)



        print("     Recommended:")



        for x in top_items[:3]:

            print("        %s" % x)

bg =  pd.read_csv("/kaggle/input/book-recommendation-sysguide/books.csv")
sample_recommendation(model, bg, [45, 4652, 6532])
model = LightFM(loss='warp',

                item_alpha=ITEM_ALPHA,

                no_components=NUM_COMPONENTS)



# Fit the hybrid model. Note that this time, we pass

# in the item features matrix.

model = model.fit(train,

                item_features=item_features,

                epochs=NUM_EPOCHS,

                num_threads=NUM_THREADS)
import pandas as pd

UpdatedData = pd.read_csv("../input/UpdatedData.csv")