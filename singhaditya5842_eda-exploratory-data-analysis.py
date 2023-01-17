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
df1 = pd.read_csv("../input/lish-moa/train_features.csv")

df2 = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

df3 = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")



print("shape of train_features.csv: ", df1.shape)

print("shape of train_targets_scored.csv: ", df2.shape)

print("shape of train_targets_nonscored.csv: ", df3.shape)



print("Total number of datapoints: {:,}".format(df1.shape[0]))



df1.head()
df1.describe()
# knowing the data types of each column of `train_features.csv`

data_types = df1.dtypes

unique_dtypes = data_types.unique()

print("number of dtypes in `train_features.csv`: ", len(unique_dtypes),

      "\nAnd these are: ", unique_dtypes)



Obj   = []

Int   = []

Float = []

for col, data_type in zip(df1.columns, data_types):

    if data_type == 'object':Obj.append(col)        

    elif data_type == 'int64':Int.append(col)

    elif data_type == 'float64':Float.append(col)

print("number of object data type: ", len(Obj))

print("number of int64 data type: ", len(Int))

print("number of float64 data type: ", len(Float))



assert len(Obj)+len(Int)+len(Float) == df1.shape[1]
Obj, Int
print("Number of unique values in `cp_type` col is: {} and these are: {}"

      .format(len(df1.loc[:, "cp_type"].unique()), df1.loc[:, "cp_type"].unique()))



print("Number of unique values in `cp_dose` col is: {} and these are: {}"

      .format(len(df1.loc[:, "cp_dose"].unique()), df1.loc[:, "cp_dose"].unique()))



print("Number of unique values in `cp_time` col is: {} and these are: {}"

      .format(len(df1.loc[:, "cp_time"].unique()), df1.loc[:, "cp_time"].unique()))
import matplotlib.pyplot as plt

import seaborn as sns
fig, axs = plt.subplots(1,3, figsize=(18,6))



fig.suptitle("Count Plot of Categorical variables", fontsize = 24)

sns.countplot(x ='cp_type', data = df1, ax = axs[0])

axs[0].set_title("For: cp_type", fontsize= 16)

sns.countplot(x ='cp_dose', data = df1, ax = axs[1])

axs[1].set_title("For: cp_dose", fontsize = 16)

axs[1].set(ylabel = '')

sns.countplot(x = 'cp_time', data = df1, ax = axs[2])

axs[2].set_title("For: cp_time", fontsize = 16)

axs[2].set(ylabel = '')



plt.show()
# cp_type, cp_time, cp_dose

def encode_cp_time(row):

    val = None

    if row == 24:val = 1

    elif row == 48:val = 2

    else:val = 3

    return val

        

df1["cp_type"] = df1["cp_type"].apply(lambda x: 0 if x=='trt_cp' else 1)

df1["cp_dose"] = df1["cp_dose"].apply(lambda x: 0 if x=='D1' else 1)

df1["cp_time"] = df1["cp_time"].apply(encode_cp_time)



df1.head()
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
X = df1.iloc[:, 1:]

x = StandardScaler().fit_transform(X)

x = pd.DataFrame(data=x, columns = X.columns)

x.head()
pca = PCA(n_components=None, svd_solver = 'full')

pca.fit_transform(x)

var = pca.explained_variance_ratio_
for i in range(1,9):

    print("Variance explained by top {} components: {}".format(i*100, var[:i*100].sum()))
corrmat = X.corr()
f, ax = plt.subplots(1,1, figsize =(18, 8))

sns.heatmap(corrmat, ax = ax)

plt.show()
cols = [col for col in corrmat.columns if col.startswith('c-')]

f, ax = plt.subplots(1,1, figsize =(18, 8))

sns.heatmap(corrmat.loc[cols, cols], ax = ax)

plt.show()
from sklearn.metrics import log_loss

def total_loss(y_true, y_pred):

    """

    y_true: numpy nd-array of shape (None , 206), None means any value

    y_pred: numpy nd-array of shape (None , 206)

    """

    losses = []

    for i in range(y_true.shape[1]):losses.append(log_loss(y_true[:,i], y_pred[:,i], eps=1e-15))

    return np.mean(losses)
df_test = pd.read_csv("../input/lish-moa/test_features.csv")

print("number of test datapoints: {:,}".format(df_test.shape[0]))



y_train = df2.iloc[:, 1:].values

y_train_pred = np.random.random_sample(y_train.shape) 

y_test_pred = np.random.random_sample((df_test.shape[0], y_train.shape[1])) 
tr_loss = total_loss(y_train, y_train_pred)

print("train loss: ", tr_loss)
test_df = pd.DataFrame(data = y_test_pred, columns = df2.columns[1:])

temp = pd.DataFrame(data=df_test.loc[:, 'sig_id'])

test_df = pd.concat([temp, test_df], ignore_index=False, axis=1)

test_df.head()
test_df.to_csv("./submission.csv",index=False)