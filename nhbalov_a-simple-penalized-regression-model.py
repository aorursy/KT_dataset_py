import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ", train_df.shape)

print("Test shape : ", test_df.shape)


train_df.head()
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()
maxy = train_df['y'][train_df['y'].index[train_df['y']<200]].max()

train_df['y'].ix[train_df['y']>maxy] = maxy



plt.figure(figsize=(12,8))

sns.distplot(train_df.y.values, bins=50, kde=False)

plt.xlabel('y value', fontsize=12)

plt.show()
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df.ix[:10,:]
missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')

missing_df
# Now let us look at the correlation coefficient of each of these variables #

x_cols = [col for col in train_df.columns if col not in ['id','y'] if train_df[col].dtype=='int64']



labels = []

values = []

for col in x_cols:

    if train_df[col].dtype=='int64':

        labels.append(col)

        values.append(np.corrcoef(train_df[col].values, train_df.y.values)[0,1])

corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')

    

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,80))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient of the variables")

#autolabel(rects)

plt.show()
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score



dicrvars = train_df.columns.values[10:]

X = train_df[dicrvars]

y = train_df["y"]

lassoreg = Lasso(alpha=0.05)

lassoreg.fit(X, y)

y_predict = lassoreg.predict(X)

r2train = r2_score(y, y_predict)

print (r2train)
ind = lassoreg.coef_>0

print (dicrvars[ind])

print (lassoreg.coef_[ind])