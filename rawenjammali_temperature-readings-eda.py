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
dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y %H:%M')



df=pd.read_csv("../input/temperature-readings-iot-devices/IOT-temp.csv",parse_dates=['noted_date'], date_parser=dateparse)

df.head(-5)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline
print("Number of rows and columns in our data set : ",df.shape)

print(df.describe(include='all'))
#print(df.loc[df['out/in']=="In",:])

a=df.groupby('out/in')

df.hist(column='temp')
sns.countplot(df['out/in'], palette='Set3')



df['out/in'].value_counts()
df['out/in'].value_counts().plot.pie()


ax = sns.scatterplot(x=df.noted_date.dt.month, y="temp", hue="out/in", data=df)

plt.gcf().set_size_inches((10, 10)) #useful line to scale our plot    

df.boxplot(column='temp',by=df.noted_date.dt.month)

plt.gcf().set_size_inches((20, 20))
df[df["out/in"] == 'Out']

out_df = df[df["out/in"] == 'Out']



dataDailyAv = df.resample('D', how = 'mean',dayfirst=True)


