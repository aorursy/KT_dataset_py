import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np

sns.set(rc={'figure.figsize':(11.7,8.27)})
coronaDF =  pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
#plot line graph for all countries

firstDF = coronaDF.sort_values(by=['Last Update']).drop_duplicates(subset=['ObservationDate','Province/State','Country/Region'],keep='last')
firstDF = firstDF.groupby(['ObservationDate','Country/Region']).sum()
secondDF = firstDF.unstack()
tf = secondDF['Confirmed']
plt = sns.lineplot(data = tf[['Mainland China','India','Italy']])

xticks = np.arange(0,52,4)

xlabels = [i.rsplit('/',1)[0] for i in tf.index[xticks]]

plt.set_xticks(xticks)

plt.set_xticklabels(xlabels)
plt = sns.lineplot(data = tf[['India']])

xticks = np.arange(0,52,4)

xlabels = [i.rsplit('/',1)[0] for i in tf.index[xticks]]

plt.set_xticks(xticks)

plt.set_xticklabels(xlabels)