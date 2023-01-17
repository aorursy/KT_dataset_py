# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from collections import Counter
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
#remove the ID field from the train data
train = train.drop('ID', axis = 1)
test = test.drop('ID', axis=1)
#check columns of train data
train.columns
train.info()
#We have no categorical variables.
train.select_dtypes(include=['object']).dtypes
test.shape
train.shape
train.head(10)
train.isnull().sum().any()
test.isnull().sum().any()
train['target'].describe()
#plot a distribution plot to see the distribution of the target field
plt.figure(figsize=(8,5))
sns.distplot(train['target'])
plt.figure(figsize=(8,5))
sns.distplot(np.log1p(train['target']), kde='False')
np.log1p(train['target']).describe()
plt.figure(figsize=(5,5))
plt.scatter(np.log(train['48df886f9']),np.log(train['target']))
plt.xlabel('48df886f9')
plt.ylabel('Target')
train['target'].sort_values(ascending=False)
Counter(train['target']).most_common()
#Plot a boxplot
plt.figure(figsize=(8,8))
sns.boxplot(train['target'], orient='v')
#separate the x and y variables for the train and test data
#taking the log of the target variable as it is not well distributed.
x_train = train.iloc[:,train.columns!='target']
y_train = np.log1p(train.iloc[:,train.columns=='target'])
x_test = test
#copy the x_train, y_train, and x_test datasets
x_train_copy= x_train.copy()
x_test_copy= x_test.copy()
y_train_copy= y_train.copy()
x_train.columns
print(y_train.head(10))
print(x_train.shape)
print(y_train.shape)
x_test.shape
train.columns
drop_cols=[]
for cols in x_train.columns:
    if x_train[cols].std()==0:
        drop_cols.append(cols)
print("Number of constant columns to be dropped: ", len(drop_cols))
print(drop_cols)
x_train.drop(drop_cols,axis=1, inplace = True)
drop_cols_test=[]
for cols in x_test.columns:
    if x_test[cols].std()==0:
        drop_cols_test.append(cols)
print("Number of constant columns to be dropped: ", len(drop_cols_test))
print(drop_cols_test)
x_test.drop(drop_cols,axis=1, inplace = True)
x_train.shape
x_test.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
from sklearn.decomposition import PCA
pca_x = PCA(0.95).fit(x_train)
print('%d components explain 95%% of the variation in data' % pca_x.n_components_)
pca = PCA(n_components=1527)
#fit with 1527 components on train data
pca.fit(x_train)
#transform on train data
x_train_pca = pca.transform(x_train)
#transform on test data
x_test_pca = pca.transform(x_test)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train_pca, y_train)
rf_pca_predict = rf.predict(x_test_pca)
rf_pca_predict = np.expm1(rf_pca_predict)
print(rf_pca_predict)
print(len(rf_pca_predict))
submission = pd.read_csv('../input/sample_submission.csv')
submission["target"] = rf_pca_predict
print(submission.head())
submission.to_csv('sub_PCA_LR.csv', index=False)
print(submission['target'])
from sklearn.decomposition import TruncatedSVD
svd_x = TruncatedSVD(n_components=1500,n_iter=20, random_state=42)
svd_x.fit(x_train_copy)
#code to select those components which attribute for 95% of variance in data
count = 0
for index, cumsum in enumerate(np.cumsum(svd_x.explained_variance_ratio_)):
    if cumsum <=0.95:
      count+=1  
    else:
        break
print(count)
for index, cumsum in enumerate(np.cumsum(svd_x.explained_variance_ratio_)):
    print(index, cumsum)
svd = TruncatedSVD(n_components=601, random_state=42)
#fit the TSVD on the train data
svd.fit(x_train_copy)
#transform on the x_train data
x_train_svd = svd.transform(x_train_copy)
#transform on the x_test data
x_test_svd = svd.transform(x_test_copy)
rf.fit(x_train_svd, y_train_copy)
rf_tsvd_predict = rf.predict(x_test_svd)
rf_tsvd_predict = np.expm1(rf_tsvd_predict)
print(rf_tsvd_predict)
submission = pd.read_csv('../input/sample_submission.csv')
submission["target"] = rf_tsvd_predict
print(submission.head())
submission.to_csv('sub_TSVD_LR.csv', index=False)
