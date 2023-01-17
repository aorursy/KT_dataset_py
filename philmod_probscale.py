!pip install probscale
import matplotlib.pyplot as plt

import probscale

import seaborn

clear_bkgd = {'axes.facecolor':'none', 'figure.facecolor':'none'}

seaborn.set(style='ticks', context='notebook', rc=clear_bkgd)



fig, ax = plt.subplots(figsize=(8, 4))

ax.set_ylim(1e-2, 1e2)

ax.set_yscale('log')



ax.set_xlim(0.5, 99.5)

ax.set_xscale('prob')

seaborn.despine(fig=fig)