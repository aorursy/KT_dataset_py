# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/creditcardfraud/creditcard.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
FILEPATH = '/kaggle/input/creditcardfraud/creditcard.csv'
df = pd.read_csv(FILEPATH, engine='python')



df.head(10)
df.shape
#Description of DataFrame

df.describe()
#See if missing values are present

df.isnull().values.any()
#How many types of values are present

print(pd.value_counts(df['Class']))
import matplotlib.pyplot as plt
LABELS = ["Normal", "Fraud"]
#Display plot of value counts



class_counts = pd.value_counts(df['Class'], sort = True)

class_counts.plot(kind = 'bar', rot=0)

plt.title('Counts of Fraud/Normal')

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Count")
fraud = df[df["Class"]==1]



normal = df[df["Class"]==0]
print(fraud.shape, normal.shape)
#Use describe to look at how much the range of values are in the entire DataFrames

#fraud.describe()

#For specific column add column name like so

fraud.Amount.describe()
normal.Amount.describe()
#Subplots can be used to add multiple plots in the same window

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

fig.suptitle("Amount per class", fontsize='large')

bins = 50

#Specify which axis where you are goin to plot

ax1.hist(fraud.Amount, bins = bins)

ax1.set_title('Fraudulent amount')

ax2.hist(normal.Amount, bins = bins)

ax2.set_title('Normal amount')

plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

#plt.xlim((0, 20000))

plt.yscale('log') # Used to better view the second plot

plt.show()