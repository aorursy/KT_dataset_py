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
dataset=pd.read_csv("/kaggle/input/groceries-dataset/Groceries_dataset.csv")

dataset.head()

features=['Date','itemDescription']

dataset=dataset[features]

dataset.head()
for i in range(dataset['Date'].shape[0]):

    x=dataset['Date'][i].split("-")

    dataset['Date'][i]=x[2]
import matplotlib.pyplot as plt

import seaborn as sn

plt.figure(figsize=(14,6))

dataset.groupby('Date')['itemDescription'].count().plot.bar()

#Total sales of 2014 vs 2015
#Highest sold product

plt.figure(figsize=(14,6))

dataset.groupby('itemDescription')["Date"].count().sort_values(ascending=False).head(5).plot.bar(color="orange")