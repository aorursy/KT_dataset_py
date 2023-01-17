# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import Binarizer
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
X = pd.read_csv("../input/all_train.csv")
y = X["has_parkinson"]
X.drop(["has_parkinson","id"], axis=1,inplace=True)
mean_features = X.describe().mean()
col_names = np.array(mean_features.index)
#mean_features = np.array(mean_features.values)
X_new = pd.DataFrame(index=range(len(X)),columns=col_names)
for i in col_names:
    X_new[str(i)] = Binarizer(threshold=mean_features[i]).fit_transform(X[i].values.reshape(-1, 1))
print(X_new.head())
X_new['has_parkinson'] = y
X_new.to_csv("binarized_train.csv",index=False)
