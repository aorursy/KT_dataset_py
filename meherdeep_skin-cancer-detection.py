# Pre-defined code, nothing to edit here



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from PIL import Image

np.random.seed(123)



from sklearn.metrics import confusion_matrix

import itertools



import keras

from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import backend as K



# Todo: At some point it might be useful to use another framework like Pytorch

#and see how that compares to Tensorflow.
base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')



# Merge images from both folders into one dictionary



imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}



# This dictionary is useful for displaying more human-friendly labels later on



lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}

# Read CSV file

data = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))



# Create some new columns (path to image, human-readable name) and review them



data['path'] = data['image_id'].map(imageid_path_dict.get)

data['cell_type'] = data['dx'].map(lesion_type_dict.get) 

data['cell_type_idx'] = pd.Categorical(data['cell_type']).codes

data.head(10)
data.describe()
data.describe(exclude=[np.number])
# Plot cell types



fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))

data['cell_type'].value_counts().plot(kind='bar', ax=ax1)
# Too many melanocytic nevi - let's balance it a bit!



data = data.drop(data[data.cell_type_idx == 4].iloc[:5000].index)



fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))

data['cell_type'].value_counts().plot(kind='bar', ax=ax1)
# Plotting dx_type = diagnosis type

data['dx_type'].value_counts().plot(kind='bar')

# Plotting localization of cancer



data['localization'].value_counts().plot(kind='bar')
# Plotting sex distribution



data['sex'].value_counts().plot(kind='bar')
#Reading the images through its path



data['image'] =data['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
#Printing the images



n_samples = 5

fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))

for n_axs, (type_name, type_rows) in zip(m_axs, 

                                         data.sort_values(['cell_type']).groupby('cell_type')):

    n_axs[0].set_title(type_name)

    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):

        c_ax.imshow(c_row['image'])

        c_ax.axis('off')

fig.savefig('category_samples.png', dpi=300)
