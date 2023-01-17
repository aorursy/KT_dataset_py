import numpy as np

import pandas as pd

from matplotlib import pyplot as plt, cm
data = pd.read_csv('../input/HR_comma_sep.csv')

data.describe()
fig = plt.figure(figsize=(12, 10))



x, y, c = 'last_evaluation', 'satisfaction_level', 'left'

ax = plt.subplot(111)

C = ax.scatter(data[x], data[y], c=data[c], lw=0, alpha=.2, s=30, cmap='viridis')

cb = plt.colorbar(C)

cb.set_label(c)

cb.set_alpha(1)

cb.draw_all()

ax.set_ylim(0, 1)

ax.set_xlim(0, 1)

ax.set_xlabel(x)

ax.set_ylabel(y)
fig = plt.figure(figsize=(12, 10))



x, y, c = 'last_evaluation', 'satisfaction_level', 'number_project'

ax = plt.subplot(111)

C = ax.scatter(data[x], data[y], c=data[c], lw=0, alpha=.2, s=30, cmap=cm.get_cmap('viridis', 7), vmin=.5, vmax=7.5)

cb = plt.colorbar(C)

cb.set_label(c)

cb.set_alpha(1)

cb.draw_all()

ax.set_ylim(0, 1)

ax.set_xlim(0, 1)

ax.set_xlabel(x)

ax.set_ylabel(y)
m = data['left'] == 1
for v in sorted(data['number_project'].unique()):

    mask = data['number_project'] == v

    counts = data[mask]['left'].value_counts()

    left = counts.get(1, 0)

    stayed = counts.get(0, 0)

    print(v, left / stayed)
plt.figure()

ax = plt.gca()

data.boxplot('satisfaction_level', by='number_project', ax=ax)

data['number_project'].value_counts()
plt.figure()

ax = plt.gca()

data[m].boxplot('satisfaction_level', by='number_project', ax=ax)

data[m]['number_project'].value_counts()
plt.figure()



counts1 = data[~m]['number_project'].value_counts()

plt.bar(counts1.index-.3, counts1, width=.3, color='b', label='stayed')



counts2 = data[m]['number_project'].value_counts()

plt.bar(counts2.index, counts2, width=.3, color='r', label='left')

plt.legend(loc=1)

plt.xlabel('project #')

plt.ylabel('N')

plt.xlim(1.5, 7.5)

plt.ylim(0, 1.1*max(counts1.max(), counts2.max()))