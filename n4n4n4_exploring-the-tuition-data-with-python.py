import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



df = pd.read_csv("../input/tuition_graduate.csv")

df.head()
fig = plt.figure()

ax = plt.subplot(111)



df2 = df.groupby(['school'])

cm = plt.get_cmap('gist_rainbow')

ax.set_prop_cycle(color=[cm(1 * i/9) for i in range(9)])

for i in df2:

    #i[0] is the name of the series

    #i[1] is a dataframe

    ax.plot(i[1]['academic.year'],i[1]['cost'],label=i[0])



plt.legend()



rectangle = ax.get_position()

ax.set_position([rectangle.x0, rectangle.y0, rectangle.width * 1, rectangle.height])



# Place legend

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.title("Cost of Graduate School at Harvard")

plt.show()
#undergraduate data

df = pd.read_csv("../input/undergraduate_package.csv")

df.head()

fig = plt.figure()

ax = plt.subplot(111)

df_grouped =  df.groupby(['component'])



for i in df_grouped:

    ax.plot(i[1]['academic.year'],i[1]['cost'],label=i[0])



plt.legend()



subplot = ax.get_position()

ax.set_position([subplot.x0, subplot.y0, subplot.width * 1, subplot.height])



# Place legend

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.title("Cost of Attending Harvard as an Undergraduate")

plt.show()