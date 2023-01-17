import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import matplotlib.gridspec as gridspec

import matplotlib.ticker as ticker

from glob import glob

from itertools import chain

sns.set_style('whitegrid')

%matplotlib inline

import warnings

import os

warnings.filterwarnings('ignore')
try:

    inpath = "../input/224-v2/224_v2/224_v2/" #Kaggle

    print(os.listdir(inpath))

except FileNotFoundError:

    inpath = "./" #Local

    print(os.listdir(inpath))
data = pd.read_csv(inpath + 'Data_Entry_2017.csv')

print(data.shape)

data.head()
"""

removing datapoints which having age greater than 100

"""

data = data[data['Patient Age']<100]

data.shape
"""

Removing rows with multi label

"""

data = data[~data['Finding Labels'].str.contains("\|")]

print(data.shape)

data.sample(5, random_state = 6)
"""

Select just the columns that we need

"""

data.columns
data = data[['Image Index', 'Finding Labels']]

data.shape
no_finding_data = data[data['Finding Labels']=='No Finding'] #just the images with no finding

no_finding_data.shape
desease_data = data[data['Finding Labels']!='No Finding'] #just the images with desease

desease_data.shape
"""

Check if the desease data plus the no finding data are equals to the full data frame

"""

(desease_data.shape[0] + no_finding_data.shape[0]) == data.shape[0]
"""

Reduce the data frame of no finding to be equal to the desease dataframe

"""

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html



final_no_finding_data = no_finding_data.sample(n=desease_data.shape[0], random_state=1)

final_no_finding_data.shape
"""

Concat the final no finding data frame and the desease data frame

"""

#https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

data = pd.concat(objs=[final_no_finding_data, desease_data])

"""

Check if the balanced data frame has the expected shape

"""

(final_no_finding_data.shape[0] + desease_data.shape[0], 

 (final_no_finding_data.shape[1] + desease_data.shape[1])/2) == data.shape
print(data.shape)

# Show a sample of the dataset

data.sample(5, random_state=3)
"""

Read all the paths of the images

"""

data_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join(inpath, 'images', '*.png'))}#Dict Comprehention



# Print of the Dict

for key, value in data_image_paths.items():

    print(key,':', value)

    break



print('Scans found:', len(data_image_paths), ', Total Headers', data.shape[0])
"""

Adding the path column

"""

data['path'] = data['Image Index'].map(data_image_paths.get)

data.sample(5, random_state=3)
"""

Create a np array with all the single deseases

"""

all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
"""

Use a np array with all the unique deseases and then remove the "No Finding" label

aim to have a vector full of zeros when is no finding label

"""

# list_comprehension

all_labels = np.delete(all_labels, np.where(all_labels == 'No Finding'))

print(all_labels.size)

all_labels
# Convert all_labels to list

all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
"""

Add a column for each desease

"""

for c_label in all_labels:

    if len(c_label)>1: # leave out empty labels

        # Add a column for each desease

        data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

print(data.shape)

data.sample(5, random_state=3)
"""

Use MIN_CASES to set the minimum value of one desease in the dataframe

"""

# MIN_CASES = 1000



MIN_CASES = 1000



all_labels = [c_label for c_label in all_labels if data[c_label].sum()>MIN_CASES]

print('Clean Labels ({})'.format(len(all_labels)), 

      [(c_label,int(data[c_label].sum())) for c_label in all_labels])
"""

Resample using weight

since the dataset is very unbiased, we can resample it to be a more reasonable collection

"""

# weight is 0.04 + number of findings



sample_weights = data['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2

print(sample_weights)

sample_weights /= sample_weights.sum()

print(sample_weights)





new_size = 40000 #Of the dataframe



# All of images

data = data.sample(new_size, weights=sample_weights)

data.groupby('Finding Labels').count().sort_values('Image Index',ascending=False).head(15)
"""

Compare the previous proportions

"""

no_finding_data = data[data['Finding Labels']=='No Finding']

desease_data = data[data['Finding Labels']!='No Finding']

print(no_finding_data.shape, desease_data.shape)
label_counts = data['Finding Labels'].value_counts()[:15]

label_counts
# Single Space

fig, ax1 = plt.subplots(1,1,figsize = (12, 8))

#Bars

ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

ax1.set_xticks(np.arange(len(label_counts))+0.5)

#labels

_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
data = data[['Image Index', 'Finding Labels', 'path']]

data.shape
data.info()
data.to_pickle('data-chest-x-ray-singlelabel-balanced-resample.plk')