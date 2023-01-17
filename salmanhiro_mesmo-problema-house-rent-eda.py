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



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv')
df = df.drop(['Unnamed: 0'], axis = 1)
df.head()
df['hoa'] = df['hoa'].replace('Sem info','0')

df['hoa'] = df['hoa'].replace('Incluso','0')
df['property tax'] = df['property tax'].replace('Sem info','0')

df['property tax'] = df['property tax'].replace('Incluso','0')
df['hoa'].value_counts()
df['rent amount'].value_counts()
df['property tax'].value_counts()
df['fire insurance'].value_counts()
df['total'].value_counts()
def extract_value_from(Value):

    out = Value.replace('R$', '')

    out_ = out.replace(',', '')

    out_ = float(out_)

    return out_
df['hoa'] = df['hoa'].apply(lambda x: extract_value_from(x))

df['rent amount'] = df['rent amount'].apply(lambda x: extract_value_from(x))

df['property tax'] = df['property tax'].apply(lambda x: extract_value_from(x))

df['fire insurance'] = df['fire insurance'].apply(lambda x: extract_value_from(x))

df['total'] = df['total'].apply(lambda x: extract_value_from(x))
df.head()
import matplotlib.pyplot as plt

import seaborn as sns
df.describe()
print("Skewness: ", df['total'].skew())

print("Kurtosis: ", df['total'].kurt())
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
df = df[df['total']<= 5622.5]
df.head()
df.describe()
plt.figure(figsize=(10,10))

sns.set()

sns.kdeplot(df['total'])

plt.title('Total Price KDE')
plt.figure(figsize=(10,10))

sns.set()

sns.kdeplot(df['fire insurance'], color = 'r')

plt.title('Fire Insurance Price KDE')
plt.figure(figsize=(10,10))

sns.set()

sns.kdeplot(df['property tax'], color = 'g')

plt.title('Property Tax Price KDE')
plt.figure(figsize=(10,10))

sns.set()

sns.kdeplot(df['rent amount'], color = 'c')

plt.title('Rent Price KDE')
plt.figure(figsize=(10,10))

sns.set()

sns.kdeplot(df['hoa'], color = 'y')

plt.title('HOA Price KDE')
sns.jointplot(df['total'], df['fire insurance'], kind="hex", color="r")
sns.jointplot(df['total'], df['property tax'], kind="hex", color="g")
sns.jointplot(df['total'], df['rent amount'], kind="hex", color="b")
sns.jointplot(df['total'], df['hoa'], kind="hex", color="y")
plt.figure(figsize=(10,10))

sns.boxplot(x="city", y="total", palette=["m", "g"], data=df)

plt.title('City and Total Price')
plt.figure(figsize=(10,10))

sns.boxplot(x="rooms", y="total", palette=["r", "b"], data=df)

plt.title('Room and Total Price')
df.head()
plt.figure(figsize=(10,10))

sns.boxplot(x="bathroom", y="total", palette=["c", "gold"], data=df)

plt.title('Bathroom and Total Price')
plt.figure(figsize=(10,10))

sns.boxplot(x="parking spaces", y="total", palette=["m", "silver"], data=df)

plt.title('Parking Spaces and Total Price')
plt.figure(figsize=(10,10))

sns.boxplot(x="floor", y="total", palette=["m", "silver"], data=df)

plt.title('Floor and Total Price')
plt.figure(figsize=(10,10))

sns.boxplot(x="animal", y="total", palette=["m", "silver"], data=df)

plt.title('Animal and Total Price')
cor = df.corr()

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(cor, annot=True, square=True);