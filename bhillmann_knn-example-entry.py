import pandas as pd



import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

import sklearn
# list evaluation metrics for cross_val scoring

sorted(sklearn.metrics.SCORERS.keys())
# load up the train and test data

X = pd.read_csv("../input/csci5461spring2020/X_train.csv", index_col=0)

y = pd.read_csv("../input/csci5461spring2020/y_train.csv", index_col=0)



# standardize the data

def standardize(x):

    return (x - np.mean(x)) / np.std(x)

# apply rowwise

X = X.apply(standardize, axis=1)



# implement a KNN classifier

clf = KNeighborsClassifier(n_neighbors=5)



# run five fold cross-validation (to get sense of what our scores will be on the test data)

scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')

scores
# fit the classifier

clf.fit(X, y)
# Read in the test data

X_test = pd.read_csv("../input/csci5461spring2020/X_test.csv", index_col=0)





# apply rowwise

X_test = X_test.apply(standardize, axis=1)



# predict the test data

results = clf.predict_proba(X_test)
# inspect the results

len(results)
# need to transfrom the results to prediction

# clf returns tuple, one vector of tuples for each class

# clf.predict_proba shape -> num_class X num_samples

# clf.predict_proba tuple -> (probability_0, probability_1)

mat = np.zeros(shape=(X_test.shape[0], 200))

for ix, i in enumerate(results):

    for jx, j in enumerate(results[ix]):

        mat[jx, ix] = j[1]
# Extract the sample and class names from the test submission

y_test_sample = pd.read_csv("../input/csci5461spring2020/y_test_sample.csv", index_col=0)



# build the dataframe with proper index and column names

df_results = pd.DataFrame(data=mat, index=y_test_sample.index, columns=y_test_sample.columns)



# save to a file for submission

df_results.to_csv("/kaggle/working/knn_results.csv")