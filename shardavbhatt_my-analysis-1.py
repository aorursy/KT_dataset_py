import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import datetime
date = str(np.array(datetime.datetime.now()))



data = pd.read_csv('/kaggle/input/covid19-comparison-using-pie-charts/data.csv')



d = data.values



row = np.zeros((d.shape[0],d.shape[1]-2))

for i in range(d.shape[0]):

    row[i] = d[i,1:-1]
def func(pct, allvals):

    absolute = int(round(pct/100.*np.sum(allvals)))

    return "{:.1f}% ({:d})".format(pct, absolute)
plt.close('all')

date = str(np.array(datetime.datetime.now()))

labels = 'Infected', 'Recovered', 'Died'

fs = 20

C = ['lightskyblue','lightgreen','orange']



def my_plot(i):

    fig, axs = plt.subplots()

    axs.pie(row[i], autopct=lambda pct: func(pct, row[i]), explode=(0, 0.1, 0), textprops=dict(color="k", size=fs-2), colors = C, radius=1.5)

    axs.legend(labels, fontsize = fs-4, bbox_to_anchor=(1.1,1))

    figure_title = str(d[i,0])+': '+str(d[i,-1])+' cases on '+date

    plt.text(1, 1.2, figure_title, horizontalalignment='center', fontsize=fs, transform = axs.transAxes)

    plt.show()

    print('\n')
for i in range(4):

    my_plot(i)
for i in range(4,d.shape[0]):

    my_plot(i)