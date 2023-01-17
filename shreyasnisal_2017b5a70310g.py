# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.linear_model import LogisticRegression

data_train = pd.read_csv("../input/minor-project-2020/train.csv")

data_test = pd.read_csv("../input/minor-project-2020/test.csv")
# get features and predictions from training data

Y_train = data_train["target"]

X_train = data_train.drop(["id", "target"], axis=1)
# train logistic regression model with training data

lr = LogisticRegression(max_iter = 1500, class_weight = "balanced")

lr.fit(X_train, Y_train)
# run on test data

X_test = data_test.drop(['id'], axis = 1)

Y_pred = lr.predict(X_test)
# columns for final dataframe

id_col = data_test["id"]

id_col = id_col[:, np.newaxis].astype(int)

Y_pred = Y_pred[:, np.newaxis].astype(int)
# create final dataframe and generate csv

final_data = np.concatenate([id_col, Y_pred], axis = 1)

final_dataframe = pd.DataFrame(data = final_data, columns = ["id","target"])

final_dataframe.to_csv("2017B5A7PS0310G.csv")