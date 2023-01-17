# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
non_numeric = df.dtypes[df.dtypes == 'object'].index

numeric = df.dtypes[df.dtypes != 'object'].index
import matplotlib.pyplot as plt
import seaborn as sns

corrmat = df[numeric].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1, square=True)

#PCA on numeric data

#standardizing the features to be fed into PCA





df_numeric = df_numeric.dropna()

from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(df_numeric)

df_std = std_scale.transform(df_numeric)
df_numeric = df[numeric].drop('SalePrice', axis=1)

df_numeric[df_numeric.LotFrontage.isnull()]