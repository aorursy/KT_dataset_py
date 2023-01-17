from IPython.display import clear_output
!pip install git+https://github.com/jordipons/musicnn.git

clear_output()
file_name = '../input/audios/joram-moments_of_clarity-08-solipsism-59-88.mp3'
from musicnn.extractor import extractor

taggram, tags, features = extractor(file_name, model='MTT_musicnn', extract_features=True)
list(features.keys())
import numpy as np

frontend_features = np.concatenate([features['temporal'], features['timbral']], axis=1)
%matplotlib inline

import matplotlib.pylab as plt

import matplotlib.gridspec as gridspec



def depict_features(features, coordinates, title, aspect='auto', xlabel=True, fontsize=13):

    # plot features in coordinates

    ax = plt.subplot(coordinates) 

    plt.imshow(features.T, interpolation=None, aspect=aspect)

    # set title

    ax.title.set_text(title + ' (' + str(features.shape[1]) + ')' )

    ax.title.set_fontsize(fontsize)

    # y-axis

    ax.get_yaxis().set_visible(False)

    # x-axis

    x_label = np.arange(0, features.shape[0], features.shape[0]//5)

    ax.set_xticks(x_label)

    ax.set_xticklabels(x_label, fontsize=fontsize)

    if xlabel:

        ax.set_xlabel('(time frames)', fontsize=fontsize)
gs = gridspec.GridSpec(1, 1) # create a figure having 1 rows and 1 cols.

depict_features(features=features['timbral'],

                coordinates=gs[0, 0],

                title='timbral features',

                aspect='auto')

plt.show()
gs = gridspec.GridSpec(1, 1) # create a figure having 1 rows and 3 cols.

depict_features(features=features['temporal'],

                coordinates=gs[0, 0],

                title='temporal features',

                aspect='equal')

plt.show()
gs = gridspec.GridSpec(1, 1) # create a figure having 1 rows and 3 cols.

depict_features(features=frontend_features,

                coordinates=gs[0, 0],

                title='front-end features',

                aspect='equal')

plt.show()
gs = gridspec.GridSpec(3, 1) # create a figure having 1 rows and 3 cols.



depict_features(features=features['cnn1'],

                coordinates=gs[0, 0],

                title='cnn1 features',

                xlabel=False)



depict_features(features=features['cnn2'],

                coordinates=gs[1, 0],

                title='cnn2 features',

                xlabel=False)



depict_features(features=features['cnn3'],

                coordinates=gs[2, 0],

                title='cnn3 features')



plt.tight_layout()

plt.show()
plt.rcParams["figure.figsize"] = (9,6)

gs = gridspec.GridSpec(4, 3) # create a figure having 1 rows and 3 cols.



depict_features(features=features['mean_pool'],

                coordinates=gs[:, 0],

                title='mean-pool features')



depict_features(features=features['max_pool'],

                coordinates=gs[:, 1],

                title='max-pool features')



depict_features(features=features['penultimate'],

                coordinates=gs[3, 2],

                title='penultimate-layer features')



plt.tight_layout()

plt.show()
in_length = 3 # seconds -- by default, the model takes inputs of 3 seconds with no overlap



# depict taggram

plt.rcParams["figure.figsize"] = (10,8)

fontsize=12

fig, ax = plt.subplots()

ax.imshow(taggram.T, interpolation=None, aspect="auto")



# title

ax.title.set_text('Taggram')

ax.title.set_fontsize(fontsize)



# x-axis title

ax.set_xlabel('(seconds)', fontsize=fontsize)



# y-axis

y_pos = np.arange(len(tags))

ax.set_yticks(y_pos)

ax.set_yticklabels(tags, fontsize=fontsize-1)



# x-axis

x_pos = np.arange(taggram.shape[0])

x_label = np.arange(in_length/2, in_length*taggram.shape[0], 3)

ax.set_xticks(x_pos)

ax.set_xticklabels(x_label, fontsize=fontsize)



plt.show()