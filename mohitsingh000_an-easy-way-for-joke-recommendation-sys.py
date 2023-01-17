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
import pandas as pd

jokes = pd.read_csv("../input/jokes.csv")

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
import csv

train_jokes = csv.DictReader(open("/kaggle/input/train.csv"), delimiter=",")



# rating1 = pd.read_csv('/kaggle/input/ratings.csv')

# book = pd.read_csv('/kaggle/input/books.csv')

jokes_d = csv.DictReader(open("/kaggle/input/jokes.csv"), delimiter=",")

train.sample(9)
trainn = train.drop([train.columns[0]], axis=1)
trainn
jokes.head()
test.head()
from lightfm.data import Dataset



dataset = Dataset()

dataset.fit((trainn['user_id'].values  ),

            (trainn['joke_id'].values ),

           )
num_users, num_items = dataset.interactions_shape()

print('Num users: {}, num_items {}.'.format(num_users, num_items))

dataset.fit_partial(items=(x['user_id'] for x in train_jokes),

                    item_features=(x['joke_id'] for x in train_jokes))
(interactions, weights) = dataset.build_interactions(([tuple(i) for i in trainn.drop(['Rating'], axis = 1).values]))



print(repr(interactions))
item_features = dataset.build_item_features(((x['user_id'], [x['joke_id']])

                                              for x in train_jokes))

print(repr(item_features))
from lightfm import LightFM
model = LightFM(loss='bpr')

model.fit(interactions)
from lightfm.evaluation import precision_at_k
print("Train precision: %.2f" % precision_at_k(model,interactions, k=5).mean())
def sample_recommendation(model, data, user_ids):





    n_users, n_items = interactions.shape



    for user_id in user_ids:

        known_positives = jokes['joke_text'][interactions.tocsr()[user_id].indices]



        scores = model.predict(user_id, np.arange(n_items))

        top_items = jokes['joke_text'][np.argsort(-scores)]



        print("User %s" % user_id)

        print("     Known positives:")



        for x in known_positives[:3]:

            print("        %s" % x)



        print("     Recommended:")



        for x in top_items[:3]:

            print("        %s" % x)

sample_recommendation(model, train, [45, 122, 100])