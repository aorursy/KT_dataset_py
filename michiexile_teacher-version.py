# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import scipy as sp

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
from sklearn import preprocessing, pipeline, ensemble, utils

from sklearn import model_selection, svm, linear_model



model = svm.SVC(kernel="poly", degree=4)



X = train.drop(["Id", "Category"], axis=1)

y = train["Category"]



X_t, y_t = utils.resample(X, y, n_samples=20000) 
model.fit(X_t, y_t)
y_pred = model.predict(test.drop(["Id"], axis=1))

submission = pd.DataFrame(

{

    "Id": test["Id"],

    "Category": y_pred

})

submission.to_csv("submission.csv", index=False)
