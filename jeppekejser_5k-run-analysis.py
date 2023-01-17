import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../input/activity_2811497722.csv')

data
data['Sekunder'] = data['Tid'].str.split(':').apply(lambda x: int(x[0]) * 60 * 60 + int(x[1]) * 60 + float(x[2]))
time_by_split = data.groupby('Split').mean()[['Sekunder']]

time_by_split = time_by_split.drop(['6', 'Summary'])
candence_by_split = data.groupby('Split').mean()[['Gennemsnit kadence løb']]

candence_by_split  = candence_by_split.drop(['6', 'Summary'])
candence_by_sec = data.groupby('Sekunder').mean()[['Gennemsnit kadence løb']]

candence_by_sec = candence_by_sec.drop([11.593, 983.903])


sns.despine(bottom=True, left=True)
fig, axarr = plt.subplots(2, 2, figsize=(25, 25))

time_by_split.plot.line(figsize=(25, 25), color='green',
                        fontsize=16, ax=axarr[0][0])

axarr[0][0].set_title('Tid pr. km.', fontsize=18)


candence_by_sec.plot.bar(figsize=(25, 25), color='lightgreen',
                         fontsize=16, ax=axarr[0][1])

axarr[0][1].set_title('Sekunder pr. km./kandance', fontsize=18)


ax = candence_by_split.plot.line(figsize=(25, 25), color='lightblue',
                                fontsize=16, ax=axarr[1][0],)

axarr[1][0].set_title('genmsnt. kandance pr. km.', fontsize=18)

ax = candence_by_split.plot.bar(figsize=(25, 25), color='lightblue',
                                fontsize=16, ax=axarr[1][1],)

axarr[1][1].set_title('genmsnt. kandance pr. km.', fontsize=18)


sns.despine(bottom=True, left=True)