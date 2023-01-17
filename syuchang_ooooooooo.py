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
import os
import pandas as pd
from pandas import DataFrame as df

d=pd.read_csv("train.csv")
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg,d.iloc[:,2:784], d["label"], cv=3)
print(scores.mean())