%pip install pyreclab
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyreclab
train_dataset = pd.read_csv('/kaggle/input/copy-goodreads/train_Interactions.csv')

train_dataset.head(10)
train_dataset = train_dataset.replace(0, 1e-9)
train_dataset.to_csv(path_or_buf = 'train_dataset.csv', columns = ['userID', 'bookID', 'rating'], header = ['userid', 'itemid', 'rating'])

del train_dataset #unload memory
test_dataset = pd.read_csv('/kaggle/input/testing/pairs_Rating.txt')

test_dataset.head(10)
test_dataset[['userid', 'itemid']] = test_dataset['userID-bookID'].str.split('-', expand = True)

test_dataset.head(10)
model = pyreclab.SlopeOne(dataset = '/kaggle/working/train_dataset.csv', dlmchar = b',',header = False,usercol = 1,itemcol = 2,ratingcol = 3 )

model.train(progress=True)
predictions = []

for index, row in test_dataset.iterrows():

    try:

        pred = model.predict(row['userid'], row['itemid'])

    except:

        pred = 5.0 #there is an issue with user ID existing in test dataset but not in train dataset, this is simply to handle the problem temporarily

    predictions.append([row['userID-bookID'], pred])
df = pd.DataFrame(predictions, columns = ['userID-bookID', 'prediction'])

df.head(10)
df.to_csv(path_or_buf = 'slope_one_submission.txt', index = False)

del model
model = pyreclab.SVD(dataset = '/kaggle/working/train_dataset.csv', dlmchar = b',',header = False,usercol = 1,itemcol = 2,ratingcol = 3 )

model.train(progress=True)
predictions = []

for index, row in test_dataset.iterrows():

    try:

        pred = model.predict(row['userid'], row['itemid'])

    except:

        pred = 5.0 #there is an issue with user ID existing in test dataset but not in train dataset, this is simply to handle the problem temporarily

    predictions.append([row['userID-bookID'], pred])
df = pd.DataFrame(predictions, columns = ['userID-bookID', 'prediction'])

df.head(10)
df.to_csv(path_or_buf = 'SVD_submission.txt', index = False)

del model
model = pyreclab.UserKnn(dataset = '/kaggle/working/train_dataset.csv', dlmchar = b',',header = False,usercol = 1,itemcol = 2,ratingcol = 3 )

model.train(progress=True)
predictions = []

for index, row in test_dataset.iterrows():

    try:

        pred = model.predict(row['userid'], row['itemid'])

    except:

        pred = 5.0 #there is an issue with user ID existing in test dataset but not in train dataset, this is simply to handle the problem temporarily

    predictions.append([row['userID-bookID'], pred])
df = pd.DataFrame(predictions, columns = ['userID-bookID', 'prediction'])

df.head(10)
df.to_csv(path_or_buf = 'user_knn_submission.txt', index = False)

del model
model = pyreclab.ItemKnn(dataset = '/kaggle/working/train_dataset.csv', dlmchar = b',',header = False,usercol = 1,itemcol = 2,ratingcol = 3 )

model.train(progress=True)
predictions = []

for index, row in test_dataset.iterrows():

    try:

        pred = model.predict(row['userid'], row['itemid'])

    except:

        pred = 5.0 #there is an issue with user ID existing in test dataset but not in train dataset, this is simply to handle the problem temporarily

    predictions.append([row['userID-bookID'], pred])
df = pd.DataFrame(predictions, columns = ['userID-bookID', 'prediction'])

df.head(10)
df.to_csv(path_or_buf = 'item_knn_submission.txt', index = False)

del model