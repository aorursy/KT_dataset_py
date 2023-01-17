# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf 

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 

from sklearn.metrics import confusion_matrix 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/creditcard.csv")
df.head()
df.describe()
df.isnull().sum()
print("Fraud")

print(df.Time[df.Class == 1].describe())

print()

print("Normal")

print(df.Time[df.Class == 0].describe())
f,(ax1, ax2) = plt.subplots(2,1,sharex=True, figsize=(10,4))

ax1.hist(df.Time[df.Class == 1])

ax1.set_title('Fraud')



ax2.hist(df.Time[df.Class == 0])

ax2.set_title('Normal')



plt.xlabel('Time (in Seconds)')

plt.ylabel('Number of Transactions')

plt.show()
print("Fraud")

print(df.Amount[df.Class == 1].describe())

print()

print("Normal")

print(df.Amount[df.Class == 0].describe())

plt.figure(figsize=(10,4))

plt.subplot(211)

plt.hist(df.Amount[df.Class == 1])

plt.yscale('log')

plt.title('Fraud')

plt.grid(True)
plt.subplot(212)