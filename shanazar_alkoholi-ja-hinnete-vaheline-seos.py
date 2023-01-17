import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline

pd.set_option('display.max_rows', 20)

df = pd.read_csv("../input/student-mat.csv")
list = []

#Ehksiis liidame nädala sees ja nädalavahetusel tarbitava alkoholi kokku

df['Dalc'] = df['Dalc'] + df['Walc']

for i in range(11):

    list.append(len(df[df.Dalc == i]))

ax = sns.barplot(x = [0,1,2,3,4,5,6,7,8,9,10], y = list)

plt.ylabel('Õpilaste arv')

plt.xlabel('Nädalane alkoholitarbimine')
import matplotlib.pyplot as plt

df['Dalc'] = df['Dalc'] - df['Walc']

df.Walc.plot.hist(bins=5, stacked=True, alpha=0.5);

df.Dalc.plot.hist(bins=5, stacked=True, alpha=0.5);

plt.legend(["Alkoholitarbimine nädalavahetuseti","Alkoholitarbimine nädala sees"])
keskmine = sum(df.G3)/float(len(df))

df['Dalc'] = df['Dalc'] + df['Walc']

df['Keskmine'] = keskmine

df['Keskmine'] = ['Üle keskmise' if i > keskmine else 'Alla keskmise' for i in df.G3]

sns.swarmplot(x=df['Dalc'], y = df['G3'], hue = df['Keskmine'],palette={'Üle keskmise':'lime', 'Alla keskmise': 'red'})
df.groupby(["Walc", "Dalc"])["G1", "G2","G3"].mean().round(2)
df.to_csv("tulemus.csv")