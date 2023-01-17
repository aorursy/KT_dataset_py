

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import euclidean_distances



from sklearn.utils import resample



def main():

    #docs = check_output(["ls", "../input"]).decode("utf8")

    clicks_train = pd.read_csv("../input/clicks_train.csv")

    print("Imported training data")

    clicks_test = pd.read_csv("../input/clicks_test.csv")

    print("Imported testing data")

    

    train_X = clicks_train [["display_id", "ad_id"]].DataFrame.as_matrix

    train_y = clicks_train [["clicked"]].DataFrame.as_matrix

    

    test_X = clicks_test [["display_id", "ad_id"]].DataFrame.as_matrix

    test_y = clicks_test [["clicked"]].DataFrame.as_matrix

    

    h = HyperForest ()

    h.fit ()



# Any results you write to the current directory are saved as output.
class HyperForest (BaseEstimator, ClassifierMixin):

    trees = []

    def __init__(self):

        pass

    

    def fit(self, X, y):

        X, y = check_X_y (X, y)

        for _ in range(100):

            X_b, y_b = resample(X, y, replace=True)

            

            h = HyperTree ()

            h.fit (X_b, y_b)

            trees += [h]
    def __init__(self, demo_param='demo'):

        self.demo_param = demo_param

    

    def fit(self, X, y):

        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)

        return self

    

    def predict (self, X):

        return self.classes_ [0]
if __name__ == "__main__": main()