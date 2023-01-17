import xgboost as xgb

import numpy as np 

import pandas as pd
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print ("Train data shape:", train.shape)

print ("Test data shape:", test.shape)
test = (test - 128.0) / 128.0
y = train.label

X = (train.drop(['label'], axis=1) - 128.0) / 128.0
from sklearn.model_selection import train_test_split

X_train, X_eval, Y_train, Y_eval = train_test_split(X, y, test_size = 0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=Y_train)

deval = xgb.DMatrix(X_eval, label=Y_eval)
param = {

    'max_depth': 10,  # the maximum depth of each tree

    'eta': 0.3,  # the training step for each iteration

    'silent': 1,  # logging mode - quiet

    'objective': 'multi:softprob',  # error evaluation for multiclass training

    'num_class': 10}  # the number of classes that exist in this datset

num_round = 100  # the number of training iterations
bst = xgb.train(param, dtrain, num_round)
from sklearn.metrics import accuracy_score

prob = bst.predict(deval)

pred_train = pd.DataFrame(np.asarray([np.argmax(line) for line in prob]))

print('Validation Set Accuracy:', accuracy_score(Y_eval, pred_train))

# Validation Set Accuracy: 0.970595238095 (last checked)
preds = bst.predict(xgb.DMatrix(test))

predictions = pd.DataFrame(np.asarray([np.argmax(line) for line in preds]))
submission=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions.values.ravel()})
submission.head()
submission.to_csv('submission.csv', index=False)