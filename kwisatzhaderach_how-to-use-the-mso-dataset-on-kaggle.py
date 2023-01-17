import h5py

import numpy as np

import pandas as pd



def load_data(base_path):

    with h5py.File(base_path + 'imgIdx.mat','r') as f:

        data = [["".join([chr(val) for val in f[reference][()]])

                     for reference in f['imgIdx']['name'][:,0]],

               [[val[0] for val in f[reference][()]]

                  for reference in f['imgIdx']['label'][:,0]],

               [np.array(f[reference][()])

                for reference in f['imgIdx']['anno'][:,0]]]

    df = pd.DataFrame(np.column_stack(data),

                      columns=['filename','label','annotation'])

    df['annotation'] = df['annotation'].apply(lambda x:x.T.astype(int))

    return df



df = load_data('/kaggle/input/mso-dataset/')

df.sample(5)
import cv2

%matplotlib inline

from matplotlib import pyplot as plt



img_path ='/kaggle/input/mso-dataset/MSO_img/MSO_img/'

fig,axes = plt.subplots(2,3,figsize=(16,9))

for i,ax,ex in zip(range(6),axes.flatten(),df.loc[df['label'] > 1].sample(6).to_numpy()):

    im = cv2.cvtColor(cv2.imread(img_path + ex[0]), cv2.COLOR_BGR2RGB)

    for j,box in enumerate(ex[2]):

        cv2.rectangle(im,(box[0],box[1]),(box[2],box[3]),(255,22,22),3)

    ax.imshow(im)