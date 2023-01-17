%reset -f
import pandas as pd

from time import time
import numpy as np

import os
# 1.4 Model building

#     Install h2o as: conda install -c h2oai h2o=3.22.1.2

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator



# 1.5 for ROC graphs & metrics

import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics

pd.options.display.max_columns = 300
os.chdir("../input")

print(os.listdir())
train=pd.read_csv("fashion-mnist_train.csv")

test=pd.read_csv("fashion-mnist_test.csv")

train.head()
train.shape
train.dtypes.value_counts() 

# 13.1 Start h2o

h2o.init()
X = h2o.H2OFrame(train)

X.shape
X_columns = X.columns[1:785] 
X_columns
Y_columns = X.columns[0]
Y_columns
dl_model = H2ODeepLearningEstimator(epochs=1000,

                                    distribution = 'bernoulli',                 # Response has two levels

                                    missing_values_handling = "MeanImputation", # Not needed by us

                                    variable_importances=True,

                                    nfolds = 2,                           # CV folds

                                    fold_assignment = "Stratified",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [30,30],                  # ## more hidden layers -> more complex interactions

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')
X['label'] = X['label'].asfactor()
start = time()

dl_model.train(X_columns,

               Y_columns,

               training_frame = X)

end = time()

(end - start)/60

X_test = h2o.H2OFrame(test)
X_test['label'] = X_test['label'].asfactor()

result = dl_model.predict(X_test[: , 0:785])
xe=X_test['label'].as_data_frame()

xe['result']=result[0].as_data_frame()

xe.head()
out = (xe['result'] == xe['label'])
out
np.sum(out)/out.size