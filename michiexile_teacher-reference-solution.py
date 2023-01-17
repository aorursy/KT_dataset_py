# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
from sklearn import preprocessing, pipeline, ensemble

from sklearn import model_selection, svm



model = ensemble.BaggingClassifier(

    svm.SVC(kernel="rbf"),

    max_samples=10000, max_features=0.8, n_jobs=-1)



ohe = preprocessing.OneHotEncoder(categories="auto")



X = train.drop(["Id", "Cover_Type"], axis=1)

y = train["Cover_Type"]

model.fit(X, y)

model.score(X, y)
y_pred = model.predict(test.drop(["Id"], axis=1))

submission = pd.DataFrame(

{

    "Id": test["Id"],

    "Cover_Type": y_pred

})

submission.to_csv("submission.csv", index=False)
