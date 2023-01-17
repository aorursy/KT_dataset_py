import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

from time import time



from sklearn.preprocessing import StandardScaler



import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator
os.listdir("../input")
h2o.init()
## Read the dataset - both train and test using H20
train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")

test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")
train.shape
test.shape
train.head(5)
tmp_df = train.as_data_frame()
tmp_df.label.unique()
tmp_r = tmp_df.values[5,1:]

tmp_r.shape
tmp_r = tmp_r.reshape(28,28)
tmp_r.shape
plt.imshow(tmp_r)

plt.show()
y_target = train.columns[0]

y_target
X_predictors = train.columns[1:785]
train["label"] = train["label"].asfactor()
train["label"].levels()
model_h2o = H2ODeepLearningEstimator(

                distribution="multinomial",

                activation="RectifierWithDropout",

                hidden=[50,50,50],

                input_dropout_ratio=0.2,

                standardize=True,

                epochs=1000

                )
start = time()

model_h2o.train(X_predictors, y_target, training_frame= train)

end = time()

(end-start)/60
result = model_h2o.predict(test)
result.shape

result.as_data_frame().head(5)
re = result.as_data_frame()
re["predict"]
re["actual"] = test["label"].as_data_frame().values
out = (re["predict"] == re["actual"])

np.sum(out)/out.size