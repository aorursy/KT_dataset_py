import sys

!cp ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import time



import pandas as pd



import sklearn.neighbors

import sklearn.svm

import sklearn.ensemble

from sklearn.model_selection import KFold



import cudf

import cuml



import matplotlib.pyplot as plt

import numpy as np
NFOLDS = 5

ITERATION = 300
train_orig_df = pd.read_csv("../input/titanic/train.csv")

test_orig_df = pd.read_csv("../input/titanic/test.csv")
train_df = train_orig_df.copy()

train_df.drop(["Cabin", "Ticket", "Name"], axis=1, inplace=True)

train_df = pd.get_dummies(train_df.iloc[:, 1:], columns=["Pclass", "Sex", "Embarked"])

train_df.dropna(inplace=True)



X_all = train_df.drop(["Survived"], axis=1).astype("float32")

y_all = train_df["Survived"].astype("int32")



X_all_gpu = cudf.from_pandas(X_all)

y_all_gpu = cudf.from_pandas(y_all)
def bench(X, y, classifiers, params):

    elapsed = {}

    for name, clf_class in classifiers.items():

        elapsed_list = []



        for _ in range(ITERATION):

            kf = KFold(n_splits=NFOLDS)

            clf = clf_class()

            clf.set_params(**params[name])



            elapsed_sum = 0

            for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):

                X_train = X_all.iloc[train_idx]

                y_train = y_all.iloc[train_idx]

                X_val = X_all.iloc[val_idx]

                y_val = y_all.iloc[val_idx]



                start = time.time()

                clf.fit(X_train, y_train)

                elapsed_sum += time.time() - start



            elapsed_list.append(elapsed_sum)



        elapsed[name] = pd.Series(elapsed_list).mean()

    return elapsed
classifiers = {

    "KNN": sklearn.neighbors.KNeighborsClassifier,

    "SVM": sklearn.svm.SVC,

    "RandomForest": sklearn.ensemble.RandomForestClassifier

}



params = {

    "KNN": {},

    "SVM": {

        "random_state": 47

    },

    "RandomForest": {

        "n_estimators": 100,

        "random_state": 47

    }

}



elapsed_sklearn = bench(X_all, y_all, classifiers, params)
classifiers = {

    "KNN": cuml.neighbors.KNeighborsClassifier,

    "SVM": cuml.svm.SVC,

    "RandomForest": cuml.ensemble.RandomForestClassifier

}



params = {

    "KNN": {},

    "SVM": {},

    "RandomForest": {

        "n_estimators": 100

    }

}



elapsed_cuml = bench(X_all_gpu, y_all_gpu, classifiers, params)
left = np.arange(len(elapsed_sklearn.keys()))

width = 0.3



fig = plt.figure(figsize=(6, 6))

fig.patch.set_alpha(1)



plt.subplot(1, 1, 1)



plt.bar(left, elapsed_sklearn.values(), color='b', width=width, label="scikit-learn", align="center")

plt.bar(left + width, elapsed_cuml.values(), color="g", width=width, label="cuML", align="center")



plt.xticks(left + width / 2, elapsed_sklearn.keys())

plt.legend(loc=2)

plt.ylabel("sec / iter")

plt.title("fit() performance")

plt.show()