# Loading the packages

import numpy as np

import pandas as pd 

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier
# Loading the training dataset

df_train = pd.read_csv("../input/train.csv")
y = df_train["target"]

# We exclude the target and id columns from the training dataset

df_train.pop("target");

df_train.pop("id")

X = df_train 

del df_train
# Split data into training and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = DecisionTreeClassifier(class_weight='balanced')

# Good reference: 

# https://stackoverflow.com/questions/37522191/sklearn-how-to-balance-classification-using-decisiontreeclassifier
model.fit(X_train, y_train)
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus



dot_data = StringIO()



export_graphviz(model, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
y_train_predict = model.predict_proba(X_train)

y_train_predict = y_train_predict[:,1]

y_test_predict = model.predict_proba(X_test, )

y_test_predict = y_test_predict[:,1]

auc_train = roc_auc_score(y_train, y_train_predict)

auc_test = roc_auc_score(y_test, y_test_predict)

print("The AUC in the training dataset is {}".format(auc_train))

print("The AUC in the test dataset is {}".format(auc_test))
df_test = pd.read_csv("../input/test.csv")

df_test.pop("id");

X = df_test 

del df_test

y_pred = model.predict_proba(X)

y_pred = y_pred[:,1]
# submit prediction

smpsb_df = pd.read_csv("../input/sample_submission.csv")

smpsb_df.head()
smpsb_df["target"] = y_pred

smpsb_df.to_csv("decision_tree.csv", index=None)