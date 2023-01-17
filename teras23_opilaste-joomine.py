%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



pd.set_option('display.max_rows', 20)

df = pd.read_csv("../input/student-por.csv")
df
explode = (0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4)

ax = df.age.value_counts().plot.pie(figsize=(12, 12), explode=explode, fontsize=16)

ax.set_title('Õpilaste vanus', fontsize=20)

ax.set_xlabel('', fontsize=16)

ax.set_ylabel('', fontsize=16)
df.groupby("Dalc")["G3","absences","goout","Fedu","Medu","studytime"].mean()
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12,12))



a_heights, a_bins = np.histogram(df.Dalc, bins=np.arange(1, 7, 1))

b_heights, b_bins = np.histogram(df.Walc, bins=a_bins)



width = (a_bins[1] - a_bins[0]) / 3



ax.set_xticks(np.arange(1+width/2, 6, 1))

ax.set_xticklabels(np.arange(1, 6, 1))



ax.set_title('Alkoholi tarbimise kogus tööpäevadel ja nädalavahetusel', fontsize=20)

ax.set_xlabel('Alkoholi joomise kogus', fontsize=16)

ax.set_ylabel('Sagedus', fontsize=16)



work = mpatches.Patch(color='cornflowerblue', label='Tööpäev')

weekend = mpatches.Patch(color='seagreen', label='Nädalavahetus')

ax.legend(handles=[work, weekend], fontsize=12)



ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')

ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')
age = df['G3']

con = df[['Walc', 'Dalc']].mean(axis=1)

s = con.value_counts()**1.4



plt.figure(figsize=(12,12))

plt.scatter(x=age, y=con, s=s, color='cornflowerblue')

plt.title('Õpitulemuste ja joomissageduse scatterplot', fontsize=20)



plt.xlabel('Kursuse hinne skaalal 0-20', fontsize=16)

plt.ylabel('Alkoholi joomise kogus', fontsize=16)
df.plot.scatter("absences","Dalc")
pd.set_option('display.max_rows', 32)

df.groupby(["age", "Walc"]).mean().round(1)