import numpy as np

import pandas as pd
def show_feature_importances(clf, X):

    importances = clf.feature_importances_

    feature_names = X.columns

    indices = np.argsort(importances)[::-1]    



    print("Feature importances:")

    for f, idx in enumerate(indices):

        print("{:2d}. '{:5s}' ({:.4f})".format(idx, feature_names[idx], 

              importances[idx]))

              

    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), importances[indices],

           color="b", align="center")

    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices])

    plt.xlim([-1, X.shape[1]])

    plt.show()