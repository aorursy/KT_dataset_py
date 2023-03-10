# This Python 3 environment comes with many helpful analytics libraries installed.

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, f1_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory;

import datetime

import random 

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output;

print(os.getcwd())
# A list of utility functions used below to manipulate/clean data and compute accuracy metrics;



def hamming_accuracy(prediction, true_values):

    """

    Metric used in multioutput-label classification,

    for each example measures the % of correctly predicted labels.

    

    Equivalent to traditional accuracy in a single-output scenario;

    """

    return np.mean(np.sum(np.equal(prediction, true_values)) / float(true_values.size))





def get_score(prediction, true_values):    

    print("\tHamming accuracy: {:.3f}".format(hamming_accuracy(prediction, true_values)))

    print("\tAccuracy, exact matches: {:.3f}".format(accuracy_score(prediction, true_values)))

    print("\tMacro F1 Score: {:.3f}".format(f1_score(y_true=true_values, y_pred=prediction, average="macro")))

    print("\tMicro F1 Score: {:.3f}".format(f1_score(y_true=true_values, y_pred=prediction, average="micro")))

    



def build_dataframe(input_data: pd.DataFrame, col_name: str, preserve_int_col_name=False) -> pd.DataFrame:

    """

    Given an input DataFrame and a column name, return a new DataFrame in which the column has been cleaned.

    Used to transform features and labels columns from "0;1;1;0" to [0, 1, 1, 0]

    """

    vertices_dict = []

    for i, row_i in input_data.iterrows():

        features = [int(float(x)) for x in row_i[f"{col_name}s"].split(";")]

        

        new_v = {"id": i}

        for j, f in enumerate(features):

            new_v[j if preserve_int_col_name else f"{col_name}_{j}"] = f

        vertices_dict += [new_v]

    res_df = pd.DataFrame(vertices_dict)

    return res_df.set_index("id")





def bool_to_int(labels: list) -> list:

    """

    Turn a list of 0s and 1s into a list whose values are the indices of 1s.

    Used to create a valid Kaggle submission.

    E.g. [1, 0, 0, 1, 1] -> [0, 3, 4]

    """

    return [i for i, x in enumerate(labels) if x == 1]
graph_name = "ppi"

    

# Load the embeddings;

embeddings_path = "../input/embeddings.csv"

embeddings_df = pd.read_csv(embeddings_path, header=None, index_col=0)

embeddings_df.columns = ["e_" + str(col) for col in embeddings_df.columns]

# Show the first rows.

# The empty "0" row is the index name, 0 is used as placeholder if no other name is provided;

embeddings_df.head()
# Read vertex features and classes in the training set;

vertices_path = f"../input/{graph_name}_train.csv"

vertices_train = pd.read_csv(vertices_path, sep=",", index_col="id")

vertices_train["dataset"] = "train"

vertices_train.head()
# Read vertex features in the test/validation set;

vertices_path = f"../input/{graph_name}_test.csv"

vertices_test = pd.read_csv(vertices_path, sep=",", index_col="id")

vertices_test["dataset"] = "test"

vertices_test.head()
# Use a temporary dict representation to turn training set features into independent columns;

X_train_df = build_dataframe(vertices_train, "feature")

X_train_df.head()
# Do the same for the test set;

X_test_df = build_dataframe(vertices_test, "feature")

X_test_df.head()
# Create a dataset with the labels of the training set (and keep numeric columns ids);

y_train_df = build_dataframe(vertices_train, "label", preserve_int_col_name=True)

y_train_df.head()
# Add each vertex embedding to the vertices df;

X_train_df = pd.merge(X_train_df, embeddings_df, left_index=True, right_index=True, how="left")

X_test_df = pd.merge(X_test_df, embeddings_df, left_index=True, right_index=True, how="left")

X_train_df.head()
# Create a classifier for the problem;

    

# Define cross-validation.

# Note that we are not using the test set when doing cross-validation,

#  as test sets are generated by cross-validation itself.

# We can also use the test set instead of cross-validation, but the accuracy estimation is usually worse;

kfolds = KFold(n_splits=10)



seed = random.randint(0, 2**32)



# We use a Logistic trained with Stochastic Gradient Descent.

# The "OneVsRestClassifier" is a meta-classifier that fits a logistic classifier on each class, independently.

# We are using default parameters, to replicate the test

sgd = SGDClassifier(loss="log", max_iter=100, tol=1e-3)

model = OneVsRestClassifier(sgd, n_jobs=1)
#%% Test with cross-validation the specified model;

scores = cross_val_score(model, X_train_df, y_train_df, cv=kfolds, n_jobs=32, verbose=2, scoring="f1_micro")

print(f"Mean crossvalidation Micro-F1: {np.mean(scores):.3f}")
# Train on the entire training set;

model.fit(X_train_df, y_train_df)



# Printing train scores.

# Look for overfitting! This score shouldn't be much higher than the one obtained with crossvalidation;

print("Train accuracy")

y_train_pred = model.predict(X_train_df)

get_score(y_train_pred, y_train_df.values)



# Predict on the test dataset;

y_test_pred = model.predict(X_test_df)
# Assemble an output file with the predictions;



y_pred = [" ".join([str(y) for y in bool_to_int(x)]) for x in y_test_pred]

y_pred_df = pd.DataFrame(y_pred, columns=["labels"], index=X_test_df.index)

y_pred_df.to_csv(f"prediction_{datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')}.csv")