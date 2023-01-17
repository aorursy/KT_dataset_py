import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold



Xy_train = pd.read_csv("../input/cat-in-the-dat/train.csv", index_col="id")

X_train = Xy_train.drop(columns=["target"])

y_train = Xy_train["target"]

X_test = pd.read_csv("../input/cat-in-the-dat/test.csv", index_col="id")



X_comb_onehot = pd.get_dummies(pd.concat([X_train, X_test]), sparse=True, columns=X_train.columns)

X_train_sparse = X_comb_onehot.loc[y_train.index].sparse.to_coo().tocsr()

X_test_sparse = X_comb_onehot.drop(index=y_train.index).sparse.to_coo().tocsr()



lr_params = dict(solver="lbfgs", C=0.2, max_iter=5000, random_state=0)

models = [LogisticRegression(**lr_params).fit(X_train_sparse[t], y_train[t])

          for t, _ in KFold(5, random_state=0).split(X_train_sparse)]

predictions = np.average([model.predict_proba(X_test_sparse)[:, 1] for model in models], axis=0)



output = pd.DataFrame({"id": X_test.index, "target": predictions})

output.to_csv("submission.csv", index=False)
