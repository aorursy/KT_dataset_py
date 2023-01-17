# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read Dataset

df = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')



# Fill missing values with zeros

df= df.fillna(0)



# Deaths per 1M population for each continent 

plt.figure(figsize=(10,6))

sns.barplot(x=df['Continent'],y=df['Deaths/1M pop'])



# Distribution of Deaths per 1M population

plt.figure(figsize=(10,6))

sns.kdeplot(data=df['Deaths/1M pop'],shade=True)



# Distribution of Cases per 1M population

plt.figure(figsize=(10,6))

sns.kdeplot(data=df['Tot Cases/1M pop'],shade=True)



# Joint Distribution of Deaths and Cases per 1 M Population.

plt.figure(figsize=(10,6))

sns.jointplot(x=df['Tot Cases/1M pop'], y=df['Deaths/1M pop'], kind="kde")



# show the relation between tests and cases per 1M poplulation.

plt.figure(figsize=(10,6))

sns.regplot(x=df['Tests/1M pop'],y=df['Tot Cases/1M pop'])







# use the Continent as hue to detect if there is a genetic effect.

plt.figure(figsize=(10,6))

sns.lmplot(x="Tests/1M pop",y="Tot Cases/1M pop", hue="Continent", data=df)
