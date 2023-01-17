# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train_V2.csv", index_col='Id')
print(train.shape)
train.head()
train.describe()
train.isna().sum()
filt = train['winPlacePerc'].isna()
train[filt]
train = train.fillna(0) 
train.corr().style.format("{:.2%}").highlight_min()
correlations = train.corr()
sns.heatmap(correlations)

def plot_correlation_heatmap(df):
    corr = df.corr()
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    
plot_correlation_heatmap(train)
X = train['walkDistance'].values.reshape(-1,1)
X[:10]
y = train['winPlacePerc'].values
y[:10]
## Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
from sklearn.model_selection import cross_val_score
cvs_lr = cross_val_score(lr, X, y, cv=15)
cvs_lr.mean(), cvs_lr.std()
## Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
cvs_dtr = cross_val_score(dtr, X, y)
cvs_dtr.mean(), cvs_dtr.std()

dtr.fit(X,y)
test = pd.read_csv("../input/test_V2.csv", index_col='Id')
test.head()
test.isna().sum().sum()
X_test = test['walkDistance'].values.reshape(-1,1)
X_test[:10]
predictions = dtr.predict(X_test).reshape(-1,1)
dfpredictions = pd.DataFrame(predictions, index=test.index).rename(columns={0:'winPlacePerc'})
dfpredictions.head(15)
dfpredictions.to_csv('submission.csv', header=True)
