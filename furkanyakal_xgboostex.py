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
train_numeric_chunks = pd.read_csv('../input/clusters-numeric/train_numeric_with_clusters.csv', iterator=True, chunksize=250000)



pd.options.display.max_columns = None

pd.options.display.max_rows = None

pd.options.display.max_colwidth = None
def get_numeric_frame():

    for data_frame in train_numeric_chunks:

        yield data_frame



get_df_numeric = get_numeric_frame()

df_numeric = next(get_df_numeric).sample(75000)
while True:

    try:

        df_numeric = pd.concat([df_numeric, next(get_df_numeric).sample(75000)], axis=0)

    except:

        break
cl_L0_one_hot = pd.get_dummies(df_numeric.ClusterL0, prefix='ClusterL0')

cl_L1_one_hot = pd.get_dummies(df_numeric.ClusterL1, prefix='ClusterL1')

cl_L2_one_hot = pd.get_dummies(df_numeric.ClusterL2, prefix='ClusterL2')

cl_L3_one_hot = pd.get_dummies(df_numeric.ClusterL3, prefix='ClusterL3')



if cl_L0_one_hot[cl_L0_one_hot.sum(axis=1)!=1].empty:

    print("Encoding L0 is successful")

    

if cl_L1_one_hot[cl_L1_one_hot.sum(axis=1)!=1].empty:

    print("Encoding L1 is successful")



if cl_L2_one_hot[cl_L2_one_hot.sum(axis=1)!=1].empty:

    print("Encoding L2 is successful")

    

if cl_L3_one_hot[cl_L3_one_hot.sum(axis=1)!=1].empty:

    print("Encoding L3 is successful")

    

df_numeric.drop(columns=["ClusterL0", "ClusterL1", "ClusterL2", "ClusterL3"], inplace=True)
df_numeric = pd.concat([df_numeric, cl_L0_one_hot, cl_L1_one_hot, cl_L2_one_hot, cl_L3_one_hot], axis=1)

del cl_L0_one_hot, cl_L1_one_hot, cl_L2_one_hot, cl_L3_one_hot
df_numeric.sample(100)
print("Number of fail parts:{} / Total parts:{}".format(len(df_numeric[df_numeric.Response == 1]), len(df_numeric)))

print("Ratio is: {}".format(len(df_numeric[df_numeric.Response == 1])/len(df_numeric)))
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
df_numeric = df_numeric.sample(300000)



# X_id = df_numeric[['Id']]



y = df_numeric.Response.ravel()

X = df_numeric.drop(columns=["Id", "Response"]).values



del df_numeric
clf = XGBClassifier()

clf.fit(X, y)

important_indices = np.where(clf.feature_importances_>0.00005)[0]

print(important_indices)
clf = XGBClassifier(max_depth=7, base_score=0.005)

scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')

scores