# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
labels = train_data['Y']
clf = RandomForestClassifier(n_estimators=256)
clf.fit(train_data.drop('Y', axis = 1), labels)
preds = clf.predict_proba(pd.read_csv("../input/test.csv"))
sample_sub = pd.read_csv("../input/sampleSubmission.csv")
sample_sub['predicted_val'] = preds
sample_sub.head(10)
