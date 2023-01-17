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



df = pd.read_csv("/kaggle/input/seeds_dataset.txt", sep = "\t", header = None, names = ["Area", "Perimeter", "Compactness", "Length of Kernal", "Width of Kernal", "Asymmetry Coefficient", "Length of Kernal Groove", "Class"])

df.head()

# Any results you write to the current directory are saved as output.
df[df.isna().any(axis = 1)]
df["Compactness"].fillna(np.mean(df["Compactness"]), inplace = True)

df["Width of Kernal"].fillna(np.mean(df["Width of Kernal"]), inplace = True)

df["Length of Kernal Groove"].fillna(np.mean(df["Length of Kernal Groove"]), inplace = True)

df["Class"].fillna(np.mean(df["Class"]), inplace = True)
pd.Series(df["Class"]).value_counts()
df["Class"] = np.around(df["Class"]).map("{:,.0f}".format)
df = df[df["Class"] != "5"]
df[df.columns].hist(figsize = (10,10))
import matplotlib.pyplot as plt

import seaborn as sns

def plotBarChart(df,col,label):

    g = sns.FacetGrid(df, col=col)

    g.map(plt.hist, label, bins=10)



for val in ["Area", "Perimeter", "Compactness", "Length of Kernal", "Width of Kernal", "Asymmetry Coefficient", "Length of Kernal Groove"]:

    plotBarChart(df,'Class',val)   
from sklearn.model_selection import train_test_split as dsplit

from sklearn.linear_model import LogisticRegression



x = df[["Area", "Perimeter",  "Length of Kernal", "Width of Kernal", "Length of Kernal Groove"]]

y = df["Class"]



for val in ["Area", "Perimeter", "Compactness", "Length of Kernal", "Width of Kernal", "Asymmetry Coefficient", "Length of Kernal Groove"]:

    pd.Series(df[val]).values.reshape(-1,1)

x_train, x_test, y_train, y_test = dsplit(x, y, random_state = 0)

reg = LogisticRegression()

reg.fit(x_train, y_train)

predictions = reg.predict(x_test)

from sklearn.metrics import accuracy_score

#r_square = metrics.score(y_test, predicted)

print("Accuracy Score :",accuracy_score(y_test, predictions))