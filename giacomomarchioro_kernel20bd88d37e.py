# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import itertools
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.pyplot import cm
import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv(r"../input/systemsfor3dsurfacemeasurements/systemsfor3Dsurface measurements.csv")
group = dataset.groupby(dataset.Core_Technology)
groups_name = group.indices.keys()
marker = itertools.cycle(('$A$', '+', '.', 'o', '*','^','<','>','|','3','X','$p$'))

plt.rcParams['figure.figsize'] = (5.5,4)
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(dataset.Ac)

ax.set_yscale('log')
ax.set_xscale('log')
for i in group:
    ax.scatter(i[1].Accuracy_microns,
               i[1].Spatial_Sampling_microns,
                label=i[0],
                marker=next(marker),
               alpha=0.6
                )

colormap = plt.cm.nipy_spectral #gist_ncar #nipy_spectral, Set1,Paired  
colorst = [colormap(i) for i in np.linspace(0, 0.8,len(ax.collections))]
colorst = ['k','darkorange','green','fuchsia','red','royalblue','teal','blueviolet','orange','blue','slategray','hotpink']
for t,j1 in enumerate(ax.collections):
    j1.set_color(colorst[t])


ax.set_xlabel ('Approximate height accuracy (micron)')
ax.set_ylabel('Spatial sampling (micron)')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
lgd = ax.legend(fontsize='small',loc='center left', bbox_to_anchor=(1, 0.5))
#ax.legend()
ax.grid()
plt.tight_layout()
#plt.savefig('Spatial_Accuracy.pdf',bbox_extra_artists=[lgd], bbox_inches='tight')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
#ax.scatter(dataset.Ac)


ax2.set_yscale('log')
ax2.set_xscale('log')
for i in group:
    ax2.scatter(i[1].Accuracy_microns,
                i[1].area_mm2,
                label=i[0],
                marker=next(marker),
                alpha=0.5
                )

colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax2.collections))]       
colorst = ['k','darkorange','green','fuchsia','red','royalblue','teal','blueviolet','orange','blue','slategray','hotpink']

for t,j1 in enumerate(ax2.collections):
    j1.set_color(colorst[t])


ax2.set_xlabel ('Approximate height accuracy (micron)')
ax2.set_ylabel('Measurable area ($mm^{2}$)')
box2 = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
lgd = ax2.legend(fontsize='small',loc='center left', bbox_to_anchor=(1, 0.5))
#ax2.legend()
ax2.grid()
#plt.tight_layout()
#ax2.text(0.01, 0.01,'Better\nperformance',transform=ax2.transAxes)

#plt.savefig('Area_Accuracy.pdf',bbox_extra_artists=[lgd], bbox_inches='tight')
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
#ax.scatter(dataset.Ac)


ax3.set_yscale('log')
ax3.set_xscale('log')
for i in group:
    ax3.scatter(i[1].Spatial_Sampling_microns,
                i[1].area_mm2,
                label=i[0],
                marker=next(marker),
                alpha=0.5
                )

colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax3.collections))]       
colorst = ['k','darkorange','green','fuchsia','red','royalblue','teal','blueviolet','orange','blue','slategray','hotpink']

for t,j1 in enumerate(ax3.collections):
    j1.set_color(colorst[t])


ax3.set_xlabel ('Spatial Sampling (micron)')
ax3.set_ylabel('Measurable area ($mm^{2}$)')
box3 = ax3.get_position()
ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
lgd = ax3.legend(fontsize='small',loc='center left', bbox_to_anchor=(1, 0.5))
#ax2.legend()
ax3.grid()
#ax3.text(0.01, 0.9,'Better\nperformance',transform=ax3.transAxes)
#plt.tight_layout()
plt.savefig('Area_sampling.pdf',bbox_extra_artists=[lgd], bbox_inches='tight')
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
#ax.scatter(dataset.Ac)


ax4.set_yscale('log')
ax4.set_xscale('log')
for i in group:
    ax4.scatter(i[1].Accuracy_microns,
                i[1].Z_working_range_mm,
                label=i[0],
                marker=next(marker),
                alpha=0.5
                )

colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax4.collections))]       
colorst = ['k','darkorange','green','fuchsia','red','royalblue','teal','blueviolet','orange','blue','slategray','hotpink']

for t,j1 in enumerate(ax4.collections):
    j1.set_color(colorst[t])


ax4.set_xlabel ('Approximate height accuracy (micron)')
ax4.set_ylabel('Working range (mm)')
box4 = ax4.get_position()
ax4.set_position([box4.x0, box4.y0, box4.width * 0.8, box4.height])

# Put a legend to the right of the current axis
lgd4 = ax4.legend(fontsize='small',loc='center left', bbox_to_anchor=(1, 0.5))
#ax2.legend()
ax4.grid()
#ax4.text(0.01, 0.9,'Better\nperformance',transform=ax4.transAxes)

#plt.tight_layout()
#plt.savefig('Zrage_Accuracy.pdf',bbox_extra_artists=[lgd4], bbox_inches='tight')
