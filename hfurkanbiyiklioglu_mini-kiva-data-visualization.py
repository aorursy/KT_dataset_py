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
df = pd.read_csv("/kaggle/input/mini-kiva-data/kiva_data.csv")

df.head()
f ,ax = plt.subplots(figsize = (15,10))

sns.barplot(data = df,x = "country",y = "loan_amount")

plt.show()
import matplotlib.ticker as mtick



f ,ax = plt.subplots(figsize = (15,10))

sns.barplot(data = df,x = "country",y = "loan_amount")

#We adding tickson the y-axis begin with a $(units of USD). 



fmt = '${x:,.0f}'

tick = mtick.StrMethodFormatter(fmt)

ax.yaxis.set_major_formatter(tick)

plt.show()
f ,ax = plt.subplots(figsize = (15,10))

sns.barplot(data = df,x = "country",y = "loan_amount",hue="gender")

#We adding tickson the y-axis begin with a $(units of USD). 



fmt = '${x:,.0f}'

tick = mtick.StrMethodFormatter(fmt)

ax.yaxis.set_major_formatter(tick)

plt.show()
plt.figure(figsize=(16,10))

sns.boxplot(data = df,x = "country",y = "loan_amount")

plt.show()
plt.figure(figsize=(16,10))

sns.boxplot(data = df,x = "activity",y = "loan_amount")

plt.show()
plt.figure(figsize=(16,10))

sns.violinplot(data = df,x = "activity",y = "loan_amount")

plt.show()
plt.figure(figsize=(16,10))

sns.violinplot(data = df,x = "country",y = "loan_amount")

plt.show()
sns.set_palette("Spectral")

plt.figure(figsize=(18,12))

sns.violinplot(data = df,x = "country",y = "loan_amount",hue="gender")

plt.show()
sns.set_palette("Spectral")

plt.figure(figsize=(18,12))

sns.violinplot(data = df,x = "country",y = "loan_amount",hue="gender",split=True)

plt.show()