import kagglegym

import numpy as np

import pandas as pd

env = kagglegym.make()

o = env.reset()

pred=  o.target

id_set = set(pred['id'].values)

s = pred['id'].isin(id_set)

while True:

    test = o.features

    pred=  o.target

    pred['y'] = 0

    #pred['id'].isin(id_set)

    s = pd.concat([pred['id'].isin(id_set),s])

    id_set = id_set.union(pred['id'].values)

    o, reward, done, info = env.step(pred)

    if done:

        print("el fin ...", info["public_score"])

        break

    if o.features.timestamp[0] % 100 == 0:

        print(reward)

# the False value indicates the number of new instruments in all timestamp.        

print(s.value_counts())