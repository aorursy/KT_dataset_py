# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.cluster import KMeans



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/xAPI-Edu-Data.csv')

df.head()

sns.pairplot(data = df, hue = 'Class', diag_kind='kde')
sns.pairplot(data = X, hue = 'cluster', diag_kind='kde')
