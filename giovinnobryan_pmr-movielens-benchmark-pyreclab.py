%pip install pyreclab
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyreclab
train_dataset = pd.read_csv('/kaggle/input/train.csv', dtype = {'user' : 'str', 'movie' : 'str'})

train_dataset.head(10)
train_dataset.to_csv(path_or_buf = 'train_dataset.csv', columns = ['user', 'movie', 'rating'], header = ['userid', 'itemid', 'rating'])

del train_dataset #unload memory



test_dataset = pd.read_csv('/kaggle/input/test.csv', dtype = {'user' : 'str', 'movie' : 'str'})
model = pyreclab.SlopeOne(dataset = '/kaggle/working/train_dataset.csv', dlmchar = b',',header = False,usercol = 1,itemcol = 2,ratingcol = 3 )

model.train(progress=True)
predictions = []

for index, row in test_dataset.iterrows():

    try:

        pred = model.predict(row['user'], row['movie'])

    except:

        pred = 5.0 #there is an issue with user ID existing in test dataset but not in train dataset, this is simply to handle the problem temporarily

    predictions.append([row['ID'], pred])
df = pd.DataFrame(predictions, columns = ['ID', 'rating'])

df.head(10)
df.to_csv(path_or_buf = 'slope_one_submission.csv', index = False)

del model
model = pyreclab.SVD(dataset = '/kaggle/working/train_dataset.csv', dlmchar = b',',header = False,usercol = 1,itemcol = 2,ratingcol = 3 )

model.train(progress=True)
predictions = []

for index, row in test_dataset.iterrows():

    try:

        pred = model.predict(row['user'], row['movie'])

    except:

        pred = 5.0

    predictions.append([row['ID'], pred])
df = pd.DataFrame(predictions, columns = ['ID', 'rating'])

df.head(10)
df.to_csv(path_or_buf = 'SVD_submission.csv', index = False)

del model