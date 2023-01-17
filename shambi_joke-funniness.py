# Import the library

import matplotlib.pyplot as plt

import numpy as np

from matplotlib_venn import venn3, venn3_circles

 

v=venn3(subsets = (100, 70, 10, 20,10,10,2), set_labels = ('Facebook Friends', 'Looks at my Posts', 'Doesn\'t think I am a prick'))

plt.show()

# The sets that will be used

v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('Objective Funniness', 'Looks at my Posts', 'Understands my Jokes'))

 

# Custom it

v.get_patch_by_id('100').set_alpha(1.0)

v.get_patch_by_id('100').set_color('white')

v.get_label_by_id('100').set_text('Unknown')

c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')

c[0].set_lw(1.0)

c[0].set_ls('dotted')

 

# Add title and annotation

plt.annotate('But seriously, they are funny...', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-100,-100),

ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),

arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))

 

# Show it

plt.show()