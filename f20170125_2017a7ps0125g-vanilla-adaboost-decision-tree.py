#IMPORTS

import numpy as np

import pandas as pd



from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor



seed = 1234
#LOAD DATA

train_df = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')

test_df = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')

sub_df = pd.read_csv('/kaggle/input/bits-f464-l1/sampleSubmission.csv')
#GET NUMERICAL VALUES FROM DATA

y_train = train_df['label']

del(train_df['label'])



x_train = np.array(train_df.values); x_test = np.array(test_df.values)



print(x_train.shape, x_test.shape)

print(y_train.shape)
#TRAINING

model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=12), n_estimators=500, learning_rate=0.1, loss='square', random_state=seed)

model.fit(x_train, y_train)
#PREDICTIONS

y_test = model.predict(x_test)

sub_df['label'] = y_test
#CONVERT TO CSV

sub_df.to_csv('submission.csv', index=False)