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
df = pd.read_csv("../input/machine-learning-class-dataset/Mall_Customers.csv")
df.head()
df.describe()
df.drop("CustomerID", inplace = True, axis = 1)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
def plot(df,var):
    plt.figure()
    sns.barplot(x = df['Genre'], y = var, data = df)
    plt.show()
plot(df,'Annual Income (k$)')
plot(df, 'Spending Score (1-100)')
sns.relplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue="Genre",sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=df)
g = sns.PairGrid(df, diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot, colors="C0")
g.map_diag(sns.kdeplot, lw=2)
plt.figure(figsize = (18,4))
sns.pointplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue="Genre",
              data=df, dodge=.532, join=False, palette="dark",
              markers="d", scale=.75, ci=None)
sns.jointplot(x='Annual Income (k$)', y='Spending Score (1-100)',
              data=df)

