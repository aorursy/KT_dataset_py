import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import xgboost as xgb
train = pd.read_csv('../input/digit-recognizer/train.csv')

test  = pd.read_csv('../input/digit-recognizer/test.csv')

train.shape, test.shape
images = train.iloc[:,1:]

labels = train.iloc[:,:1]

images.shape, labels.shape, test.shape
train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, train_size=0.8, test_size=0.2, random_state=0)

train_images.shape, valid_images.shape
gbm = xgb.XGBClassifier()

gbm.fit(train_images.values, train_labels.values.ravel())
gbm.score(valid_images.values, valid_labels.values)
predictions = gbm.predict(test.values)
submissions = pd.DataFrame({

    "ImageId": list(range(1, len(predictions)+1)),

    "Label": predictions})

submissions.to_csv("output.csv", index=False, header=True)