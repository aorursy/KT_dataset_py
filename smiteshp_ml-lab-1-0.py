# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import resample

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

train_df = train_df.set_index("id").sort_index()

train_df
len(train_df[train_df["target"] == 0])
len(train_df[train_df["target"] == 1])
train_neg_downsampled_df = resample(train_df[train_df["target"] == 0],

                                    replace=False,

                                    n_samples=3*len(train_df[train_df["target"] == 1]),

                                    random_state = 0)
train_downsampled_df = train_neg_downsampled_df.append(train_df[train_df["target"] == 1]).sort_index()
train_downsampled_df
train_downsampled_df.describe()
'''

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in train_df.items():

    sns.boxplot(x=k, data=train_df, ax=axs[index])

    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

'''
X_train = train_downsampled_df.drop("target", axis=1)

y_train = train_downsampled_df["target"]
pca = PCA(n_components=40)

X_train_pca = pca.fit_transform(X_train)
model = RandomForestClassifier(n_jobs=-1)

model.fit(X_train_pca, y_train)
test_df = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")

test_df = test_df.set_index("id").sort_index()

test_df
X_test = test_df

X_test_pca = pca.fit_transform(X_test)
y_pred = model.predict(X_test_pca)
np.unique(y_pred, return_counts=True)
submission = X_test.reset_index()["id"]

submission = pd.DataFrame(submission)

submission["target"] = y_pred
submission
submission.to_csv("submission.csv", index=False)