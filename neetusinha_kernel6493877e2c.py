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
df = pd.read_csv('/kaggle/input/covid19-research-preprint-data/COVID-19-Preprint-Data_ver2.csv')

df.head(5)
df.shape
df.info()
df.columns
df['Abstract'][0]
def allunique(df):

    for i in df:

        print(i, '=',  df[i].unique(), end = '\n\n')

allunique(df)
df1 = df.copy()

df1.head(2)
# how many cases in each day

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

plt.figure(figsize = (10,8))

sns.countplot(x = "Date of Upload", data = df)

pd.date_range(start='2020/01/13',end = '2020/05/16')

plt.xticks(rotation=90)

plt.show()

#need to fix the xaxis so we can see the date properly
import matplotlib.dates as mdates

fig,ax = plt.subplots()

ax.xaxis.set_major_locator(mdates.WeekdayLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

sns.countplot(x = "Date of Upload", data = df)



plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,6))

plt.bar(df["Date of Upload"],  df["Title of preprint"])

plt.show()



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

dt = vectorizer.fit_transform(df['Abstract'])

print(vectorizer.get_feature_names())

sns.countplot(x = 'Authors', data = df)

plt.show()