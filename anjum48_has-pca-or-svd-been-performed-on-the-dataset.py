# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.decomposition import PCA, TruncatedSVD



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
features = 1000



# Means

m = np.random.normal(size=(features), scale=10)



# Covariance matrix.

A = np.random.normal(size=(features, features), loc=1)

cov = np.dot(A, A.transpose())



generated_raw_data = np.random.multivariate_normal(m, cov, size=10000)

generated_raw_data = pd.DataFrame(generated_raw_data)



generated_raw_data.describe()
sns.distplot(generated_raw_data[48]);
# Check if the variables are correlated

plt.scatter(generated_raw_data[48], generated_raw_data[49]);
pca = PCA(n_components=200)

generated_pca_data = pca.fit_transform(generated_raw_data)



generated_pca_data = pd.DataFrame(generated_pca_data)

generated_pca_data.describe()
sns.pairplot(generated_pca_data[list(range(10))]);
svd = TruncatedSVD(n_components=200)

generated_svd_data = svd.fit_transform(generated_raw_data)



generated_svd_data = pd.DataFrame(generated_svd_data)

generated_svd_data.describe()
sns.pairplot(generated_svd_data[list(range(10))]);
generated_raw_cat_data = generated_raw_data.round(-2)

generated_raw_cat_data.describe()
sns.distplot(generated_raw_cat_data[48]);
plt.scatter(generated_raw_cat_data[48], generated_raw_cat_data[49]);
generated_pca_cat_data = pca.fit_transform(generated_raw_cat_data)



generated_pca_cat_data = pd.DataFrame(generated_pca_cat_data)

generated_pca_cat_data.describe()
sns.pairplot(generated_pca_cat_data[list(range(10))]);
generated_svd_cat_data = svd.fit_transform(generated_raw_cat_data)



generated_svd_cat_data = pd.DataFrame(generated_svd_cat_data)

generated_svd_cat_data.describe()
sns.pairplot(generated_svd_cat_data[list(range(10))]);
train_df = pd.read_csv("../input/train.csv")

train_df.describe()
sns.pairplot(data=train_df[::20], vars=train_df.columns[2:12], hue="target");