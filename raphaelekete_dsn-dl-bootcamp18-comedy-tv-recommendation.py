import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
# Read the data file
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
from surprise import SVD, NMF, CoClustering
from surprise import KNNBasic, KNNWithMeans
from surprise import Reader, Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
#Set Rating Scale
reader = Reader(rating_scale=(-5,5))

#Load data with rating scale
data = Dataset.load_from_df(train_df[['Viewers_ID', 'Response_ID', 'Rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)
# Algorithm array

models = [
    ('SVD', SVD()),
    ('NMF', NMF()),
    ('CoClustering', CoClustering()),
    ('KNNBasic', KNNBasic()),
    ('KNNWithMeans', KNNWithMeans())
]
rmse_values = {}

for name, model in models:
    # Train the algorithm on the trainset, and predict ratings for the testset
    model.fit(trainset)
    predictions = model.test(testset)
    # Then compute RMSE
    rmse_value = accuracy.rmse(predictions, verbose=False)
    rmse_values[name] = rmse_value
    print('{} rmse score is {:4f}'.format(name, rmse_value))
# We can now use the algorithm that yields the best rmse:
algo = SVD()
algo.fit(data.build_full_trainset())
#Making recommendations
recommendations = [algo.predict(a,b)[3] for a,b in zip (test_df['Viewers_ID'], test_df['Response_ID'])]
recommendations = np.round(recommendations)
submission = pd.DataFrame({'Response_ID':test_df['Response_ID'],'Rating':recommendations}, index=None)
submission.to_csv('pred_svd3.csv',index=False)