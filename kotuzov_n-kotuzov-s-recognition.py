# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import sklearn

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



full_batch = pd.read_csv("../input/voice.csv")

full_batch["label"] = full_batch["label"].map({'male': 1, 'female': 0})

Y = full_batch["label"]

X = full_batch.drop(labels=["label"],  axis = 1)



new_x = full_batch[['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'skew', 'kurt',

       'sp.ent', 'sfm', 'mode', 'centroid', 'minfun', 'maxfun',

       'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'label']]



x_train, x_test, y_train, y_test = train_test_split(new_x, Y, random_state=0,  stratify=Y)



new_model = LogisticRegression().fit(x_train, y_train)

print("train score:", new_model.score(x_train, y_train))

print("test score: ", new_model.score(x_test, y_test))


