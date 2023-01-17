# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from numpy import percentile

import seaborn as sns
train_df = pd.read_csv("../input/mercedesbenz-greener-manufacturing/train.csv")

train_df.head()
test_df=pd.read_csv("../input/mercedesbenz-greener-manufacturing/test.csv")

test_df.head()
train_df.columns
train_df.describe()
test_df.columns
train_df.shape,test_df.shape
train_df.info()
train_df.dtypes
train_df.isna()
train_df.isna().any()
test_df.isna().any()
print(train_df.nunique())
# Categorical boolean mask

categorical_feature_mask = train_df.dtypes==object



# filter categorical columns using mask and turn it into a list

categorical_cols = train_df.columns[categorical_feature_mask].tolist()
for col in train_df[categorical_cols]:

    print(col+":",train_df[col].nunique())

    print(train_df[col].unique())
# Categorical boolean mask

categorical_feature_mask = test_df.dtypes==object



# filter categorical columns using mask and turn it into a list

categorical_cols = test_df.columns[categorical_feature_mask].tolist()
for col in test_df[categorical_cols]:

    print(col+":",test_df[col].nunique())

    print(test_df[col].unique())
# import labelencoder

from sklearn.preprocessing import LabelEncoder



# instantiate labelencoder object

le = LabelEncoder()
train_df[categorical_cols] = train_df[categorical_cols].apply(lambda col: le.fit_transform(col))

train_df[categorical_cols].head(10)
train_df.var(axis=0)
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.1)

sel.fit_transform(train_df)[:5]
train_df.head(10)
print(train_df['y'].describe())
import seaborn as sns

plt.figure(figsize=(12,6))

plt.hist(train_df['y'], bins=50, color='g')

plt.xlabel('testing time in secs')
plt.figure(figsize=(15,6))

plt.plot(train_df['y'])
from sklearn.preprocessing import StandardScaler



std = StandardScaler().fit_transform(train_df)
mean_vac = np.mean(std,axis=0)

cov_mat = (std-mean_vac).T.dot((std-mean_vac)/(std.shape[0]-1))

cov_mat
#perform Eigencomposition on Covariance

cov_mat = np.cov(std.T)

eig_vals,eig_vecs=np.linalg.eig(cov_mat)

print(eig_vals,eig_vecs)
#Decreasing EignValues

eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

for i in eig_pairs:

    print(i[0])
from sklearn.decomposition import PCA



pca = PCA(n_components =10)

pca.fit_transform(train_df)

print(pca.explained_variance_ratio_)
pca = PCA().fit(std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("No of Components")

plt.ylabel("Cumulative Explained Variance")

plt.show()
train_df.head()
sns.boxplot(x=train_df['y'])
fig, ax = plt.subplots(figsize=(16,8))

ax.scatter(train_df['X0'],train_df['y'])

ax.set_xlabel('X0')

ax.set_ylabel('time')

plt.show()
# calculate interquartile range

q25, q75 = percentile(train_df.loc[:,'y'], 25), percentile(train_df.loc[:,'y'], 75)

iqr = q75 - q25

print(iqr)
cut_off = iqr * 1.5

lower, upper = q25 - cut_off, q75 + cut_off

print(lower,upper)
# identify outliers

outliers = [x for x in train_df.loc[:,'y'] if x < lower or x > upper]

outliers
len(outliers)
outliers_deleted = [x for x in train_df.loc[:,'y'] if x >= lower or x <= upper]
outliers_deleted
merc_df=train_df.append(test_df, ignore_index=True,sort=False)
merc_df=pd.get_dummies(merc_df)
merc_df.index
train, test = merc_df[0:len(train_df)], merc_df[len(train_df):]
train.shape,test.shape
X_train_1 = train.drop(['y','ID'], axis=1)

y_train_1 = train['y']



X_test_1 = test.drop(['y','ID'], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X_train_1,y_train_1, test_size=0.30, random_state=101)
from xgboost import XGBRegressor
xgb_regressor = XGBRegressor()

xgb_regressor.fit(X_train, y_train)
y_prediction = xgb_regressor.predict(X_train)
print(y_prediction)
r2_score(y_train,y_prediction)
mean_squared_error(y_train,y_prediction)
y_prediction2 = xgb_regressor.predict(X_test)

print(y_prediction2)
r2_score(y_test,y_prediction2)
mean_squared_error(y_test,y_prediction2)