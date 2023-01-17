# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone





# subclass with BaseEstimator and choose among ClassifierMixin, RegressorMixin, ClusterMixin, TransformerMixin



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models



    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(cloned_model) for cloned_model in self.models]

        # for cloned_model in self.models:

        #     self.models_.append(clone(cloned_model))\

        self.models_ = []



        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self



    # Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.train_model(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)






