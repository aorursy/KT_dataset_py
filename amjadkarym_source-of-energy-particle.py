import pandas as pd
import numpy as np
import pandas as pd
import os
import tensorflow as tf
print(os.listdir("../input"))
from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event
for i in os.listdir('../input/'):
    print(i)
def parse_event_id(filename):
    return int(filename[5:].split('-')[0])

sample_filename = os.listdir('../input/train_1')[0]
parse_event_id(sample_filename)
train_filenames = os.listdir('../input/train_1')
train_event_ids = np.unique(sorted([parse_event_id(i) for i in train_filenames]))
print('train event ids:', train_event_ids[:10],'...', train_event_ids[-10:])
def get_by_event_id_train(id_):
    if id_ in train_event_ids:
        return sorted(np.array(train_filenames)[[id_ == parse_event_id(i) for i in train_filenames]])
    else:
        return None

# Test:
event_id = 1000
print(get_by_event_id_train(event_id))
print(get_by_event_id_train(10))

cells_df = pd.read_csv('../input/train_1/'+get_by_event_id_train(event_id)[0]) 
hits_df = pd.read_csv('../input/train_1/'+get_by_event_id_train(event_id)[1])
particles_df = pd.read_csv('../input/train_1/'+get_by_event_id_train(event_id)[2])
truth_df = pd.read_csv('../input/train_1/'+get_by_event_id_train(event_id)[3])
cells_df.head()
hits_df.tail()
particles_df.tail()
truth_df.tail()
sample_particle_id = 0
while sample_particle_id == 0:
    sample_particle_id = int(truth_df.sample()['particle_id'])
print(sample_particle_id)

data = np.array(truth_df[truth_df['particle_id']==sample_particle_id][['tx', 'ty', 'tz']])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2] , c='red', marker='o', edgecolor='k')
ax.plot(data[:,0], data[:,1], data[:,2] , '-', lw=3, alpha=0.4)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.tight_layout()
plt.show()