import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
#Setting up directories
DATA_DIR = '/kaggle/input/imet-2020-fgvc7/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
LABELS = DATA_DIR + '/labels.csv'
SAMPLE_SUB = DATA_DIR + '/sample_submission.csv'
TRAIN_LABELS = DATA_DIR + '/train.csv'

#Loading CSV files. 
labels = pd.read_csv(LABELS) 
train_labels = pd.read_csv(TRAIN_LABELS)
sub = pd.read_csv(SAMPLE_SUB)
labels.head()
sub.head()
train_labels.head()
n_train, _ = train_labels.shape
n_labels, _ = labels.shape

# Adding another column as the number of attributes per image
attrib_freq = pd.DataFrame(train_labels['id'])
attrib_freq['no_of_attribute'] = np.nan
for i in tqdm (range(n_train)):
    attrib_freq.iloc[i, 1] = int(train_labels['attribute_ids'][i].count(' ') + 1)

# Plotting no_of_attribue for the images
plt.figure()
plt.hist(attrib_freq['no_of_attribute'], color = 'blue', edgecolor = 'black',
         bins = int(attrib_freq['no_of_attribute'].max()))
plt.xlabel('no_of_attribute')
plt.ylabel('images_count')
plt.title('Count of no_of_attribute')
plt.show()
labels_custom = labels
labels_custom[['attribute_type','attribute_info']] = labels_custom.attribute_name.str.split('::',expand=True,)
labels_custom.head()
labels_custom['attribute_type'].unique()

np.array(train_labels['attribute_ids'][0].split(' '), dtype = np.int)
labels
labels.iloc[np.array(train_labels['attribute_ids'][1].split(' '), dtype = np.int), 1].tolist()
from PIL import Image
index = 2
print(labels.iloc[np.array(train_labels['attribute_ids'][index].split(' '), dtype = np.int), 1].tolist())
img = Image.open(TRAIN_DIR + train_labels['id'][index] + '.png')

img
