%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os.path
import gc
import time
import pickle
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from IPython.display import Markdown, display

def read_data(directory):
    train_dtypes = {
        'object_id': np.int32,
        'mjd': np.float64,
        'passband': np.int8,
        'flux': np.float32,
    }
    train_file_path = os.path.join(directory, 'training_set.csv')
    #print('reading {}'.format(train_file_path))
    train = pd.read_csv(train_file_path, dtype=train_dtypes, usecols=list(train_dtypes.keys()))

    train_meta_dtypes = {
        'object_id': np.int32,
        'target': np.int8,
    }
    train_meta_file_path = os.path.join(directory, 'training_set_metadata.csv')
    #print('reading {}'.format(train_meta_file_path))
    train_meta = pd.read_csv(train_meta_file_path, dtype=train_meta_dtypes, usecols=list(train_meta_dtypes.keys()))

    object_id_to_target = train_meta.set_index('object_id')['target']
    train['target'] = train['object_id'].map(object_id_to_target)
    assert (pd.isnull(train['target'])).astype(np.int32).sum() == 0

    return train
directory = '../input/'
train = read_data(directory)
classes = train['target'].unique().tolist()
passbands = [x for x in range(6)]
representatives = {}
N = 3
for class_id in classes:
    representatives[class_id] = train[train.target == class_id]['object_id'].sample(N, random_state=1685).values.tolist()
pal = sns.color_palette("hls", n_colors=6)
for class_id, object_ids in representatives.items():
    display(Markdown('# Class {}'.format(class_id)))   
    for i, object_id in enumerate(object_ids):      
        plt.figure(figsize=[12, 2])    
        #display(Markdown('## object {}'.format(object_id)))
        data = train[train.object_id == object_id]        
        ax = sns.pointplot(x="mjd", y="flux", hue="passband", data=data, palette=pal, ci=None, join=False)
#         ax.get_xaxis().set_visible(False)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        plt.legend(bbox_to_anchor=(1.05, 1.02), loc=2, borderaxespad=0., title='passband')            
        plt.tight_layout()
        plt.show()