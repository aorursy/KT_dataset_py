import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(12,5)
df = pd.read_csv("../input/Pakistan Intellectual Capital - Computer Science - Ver 1.csv",

                 encoding = "ISO-8859-1")

df.head()
df_new = df[df['Other Information'].isin(['On Study Leave', 'On study leave', 'PhD Study Leave',

                                         'On Leave'])] 

x = df_new['Teacher Name'].count()

y = df['Teacher Name'].count() - x
plt.pie([x,y], explode=(0.02, 0.09), labels=['On Leave', 'Available'], autopct='%1.1f%%',

        startangle=140)

 

plt.axis('equal')

plt.show()
df['University Currently Teaching'].value_counts()[:20].plot(kind="bar")
df_new = df[df['Year'].isin([2013, 2014, 2015, 2016, 2017])]

df_new = df_new[df_new['Terminal Degree'].isin(['PhD', 'Ph.D', 'Phd'])]

df_new['Year'].value_counts().plot(kind="bar")