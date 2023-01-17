import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



demographics = np.asarray(pd.read_csv('../input/adni_demographic_master_kaggle.csv'))



train_all = np.load('../input/imgset_1/imgset_1/img_array_train_6k_1.npy')
def get_subject(subject_id):

    scan_range = range((subject_id * 62), (subject_id * 62) + 62)

    demographic = demographics[subject_id]

    

    return demographic, scan_range
subject = get_subject(5)



print(subject[0])



plt.close('all')



plot_x = 8

plot_y = 8



plot_range = plot_x * plot_y



fig, axs = plt.subplots(plot_x,plot_y, figsize=(10, 10))

fig.subplots_adjust(hspace = .5, wspace=.001)



for ax, d in zip(axs.ravel(), range(plot_range)): ax.axis('off')



for ax, d in zip(axs.ravel(), subject[1]):

    ax.imshow(train_all[d], cmap=plt.cm.jet)

    ax.set_title(str(d%62))



plt.show()   