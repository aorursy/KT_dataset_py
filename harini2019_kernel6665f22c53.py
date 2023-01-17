# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from time import time



from sklearn.preprocessing import StandardScaler



import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator

h2o.init()
train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")

test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")
tmp_df = train.as_data_frame()

tmp_df.label.unique()
tmp_r = tmp_df.values[5,1:]


tmp_r = tmp_r.reshape(28,28)
tmp_r.shape

plt.imshow(tmp_r)

plt.show()
train.shape
train.head(10)
model_h2o = H2ODeepLearningEstimator(

                distribution="multinomial",

                activation="RectifierWithDropout",

                hidden=[50,50,50],

                input_dropout_ratio=0.2,

                standardize=True,

                epochs=1000

                )
X_predictors = train.columns[1:785]
y_target = train.columns[0]

y_target
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

result.as_data_frame().head(10)


res = result.as_data_frame()
res["predict"]
res["actual"] = test["label"].as_data_frame().values
out = (res["predict"] == res["actual"])

np.sum(out)/out.size