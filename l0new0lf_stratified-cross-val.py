import numpy as np

from sklearn import datasets

np.random.seed(123)
xs, ys = datasets.make_classification(

    n_samples  = 1000, # 20 features

    n_classes  = 2, 

    weights    = [0.99, 0.01]

)
def desc_ys(ys):

    clss, cnts = np.unique(ys, return_counts=True)

    for cls, cnt in zip(clss, cnts):

        print(f"class {cls}\t: {cnt}")
desc_ys(ys)
K=10
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
for train_ix, test_ix in kfold.split(xs, ys):



    # manipulate incdices

    train_X, test_X = xs[train_ix], xs[test_ix]

    train_y, test_y = ys[train_ix], ys[test_ix]

    

    # log

    print("="*50)

    print("Test:")

    desc_ys(test_y)

    print("Train:")

    desc_ys(train_y)