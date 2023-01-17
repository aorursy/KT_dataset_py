# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/creditcard.csv")
df.shape
df.columns
df.info()
df.head()
df.tail()
df.describe()
plt.scatter(df['Amount'], df['Class'])

plt.xlabel('Amount'); plt.ylabel('Fraud');

plt.show()
df['Class'].sum()
df_fraud = df[df['Class']==1]

df_fraud.head()
plt.hist(df_fraud['Amount'], bins=50);

plt.xlabel('Amount'); plt.ylabel('Number of Frauds');

plt.show()
plt.hist(df_fraud['Time'], bins=100);

plt.xlabel('Time'); plt.ylabel('Number of Frauds');

plt.show()
sns.pairplot(df_fraud[['V1','V2','V3','V4']])