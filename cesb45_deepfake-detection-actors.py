# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.



from PIL import Image

%matplotlib inline



import matplotlib.pyplot as plt
ROOT = '/kaggle/input/deepfake-detection-challenge-actors/actors/'



videos_actor_df = pd.read_hdf(os.path.join(ROOT, 'videos_actor.h5'))

videos_actor_df
actors = videos_actor_df['actor'].unique()

num_actors = len(actors)

nc = 5

nr = int(np.ceil(num_actors / nc))



fig, ax = plt.subplots(nrows = nr, ncols=nc, figsize = (nc * 3, nr * 3))

fig.tight_layout()

r_idx = 0

c_idx = 0

for actor in actors:

    sample_path = os.path.join(ROOT, 'samples', f'{actor[0]}_{actor[1]}.jpg')

    img = Image.open(sample_path)

    ax[r_idx][c_idx].imshow(img)

    ax[r_idx][c_idx].set_title(actor)

    

    c_idx += 1

    if c_idx >= nc:

        c_idx = 0

        r_idx += 1
import glob

ex = sorted(glob.glob(os.path.join(ROOT, 'face_swaps', 's*.jpg')))

ex_imgs = [Image.open(v) for v in ex]



ex_imgs[0]
ex_imgs[1]
ex_imgs[2]
ex_imgs[3]
ex_imgs[4]
ex_imgs[5]
ex_imgs[6]