# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/indian-candidates-for-general-election-2019/LS_2.0.csv")
data.head()
for i in data.columns:

    x=i.lower().replace(' ','_').replace('\n','_').replace('__','_')

    data=data.rename(columns={i:x})
#Gender

plt.figure(figsize = (10,8))

plt.style.use('fivethirtyeight')

sns.countplot(x = data['gender'] , hue= data['winner'])

plt.show()
plt.style.use('fivethirtyeight')

plt.figure(figsize=(17,6))

sns.countplot(x = data['age'])

plt.xticks(rotation= 90)

plt.show()
data["state"].unique()
Telangana = data[data["state"] == "Telangana"]

Assam = data[data["state"] == "Assam"]

Goa =  data[data["state"] == "Goa"]

Kerala = data[data["state"] == "Kerala"]
plt.style.use('fivethirtyeight')

plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

sns.countplot(x = Telangana['party'])



plt.subplot(1,2,2)

sns.countplot(x = Assam['party'])

plt.show()
plt.style.use('fivethirtyeight')

plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

sns.countplot(x = Goa['party'])



plt.subplot(1,2,2)

sns.countplot(x = Kerala['party'])

plt.show()
plt.style.use('fivethirtyeight')

plt.figure(figsize = (20,8))

sns.countplot(x = data["education"] , hue = data['winner'])

plt.xticks(rotation = 90)

plt.show()
winners = data[data["winner"] == 1]
winners.head()
plt.style.use("fivethirtyeight")

plt.figure(figsize = (16,8))

sns.countplot( x= winners["education"])

plt.xticks(rotation = 90)

plt.show()