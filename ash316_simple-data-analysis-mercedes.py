# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head(2)
train.shape
train.isnull().any().value_counts()  #checking if there is any null values in the dataset
train['y'].hist(bins=50)
col=['X0','X1','X2','X3','X4','X5','X6','X8']

for i in col:

    sns.boxplot(x=i,y='y',data=train)

    plt.show()
col=['X0','X1','X2','X3','X4','X5','X6','X8']

for i in col:

    sns.countplot(train[i])

    plt.show()