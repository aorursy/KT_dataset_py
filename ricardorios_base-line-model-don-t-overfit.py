# Loading the packages

import numpy as np

import pandas as pd 

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
# Loading the training dataset

df_train = pd.read_csv("../input/train.csv")

y = df_train["target"]

# We exclude the target and id columns from the training dataset

df_train.pop("target");

df_train.pop("id")

X = df_train 

del df_train
# Split data into training and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# Create dummy classifer 

dummy = DummyClassifier(strategy='most_frequent', random_state=0)



# "Train" model

dummy.fit(X_train, y_train)
y_train_predict = dummy.predict_proba(X_train)

y_train_predict = y_train_predict[:,1]

#Probabilities for the class 1 in the trainind dataset

y_test_predict = dummy.predict_proba(X_test)

y_test_predict = y_test_predict[:,1] 

#Probabilities for the class 1 in the test dataset

print(np.unique(y_train_predict))

print(np.unique(y_test_predict))

print("The model always predicts 1!!")
auc_train = roc_auc_score(y_train, y_train_predict)

auc_test = roc_auc_score(y_test, y_test_predict)
print("The AUC in the training dataset is {}".format(auc_train))

print("The AUC in the test dataset is {}".format(auc_test))
df_test = pd.read_csv("../input/test.csv")

df_test.pop("id");

X = df_test 

del df_test

y_pred = dummy.predict(X)
# submit prediction

smpsb_df = pd.read_csv("../input/sample_submission.csv")

smpsb_df.head()
smpsb_df["target"] = y_pred

smpsb_df.to_csv("base_line_model.csv", index=None)