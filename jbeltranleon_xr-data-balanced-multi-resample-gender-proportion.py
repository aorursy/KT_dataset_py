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

Store all the diferent labels in a list (even the multi labels)

"""

all_labels = np.unique(data['Finding Labels'])

# Convert all_labels to list

all_labels = [x for x in all_labels]

print(f"Len: {len(all_labels)}")

print(f"all_lables = {all_labels[0:3]}...")
"""

Generate an empy dataframe to store the data with gender proportions

"""

proportion_data = pd.DataFrame(columns=data.columns)

proportion_data.head()
"""

For each label compare the shape of male againts the female 

and generate a new DataFrame with equal proportion of gender

"""

for label in all_labels:

    print(label)

    desease = data[data["Finding Labels"] == label]

    print(f"Original shape: {desease.shape}")

    if desease.shape[0] > 1: #Should be more that one to performe the split

            m = desease[desease["Patient Gender"] == 'M']

            print(f"m = {m.shape}")

            f = desease[desease["Patient Gender"] == 'F']

            print(f"f = {f.shape}")



            if ((m.shape[0] > 1) & (f.shape[0] > 1)): #each gender should have at least one entry

                if m.shape[0] > f.shape[0]: #If male is bigger that female, reduce the length of male

                    m = m.sample(f.shape[0], random_state = 3)

                    print(f"m = {m.shape}")

                elif f.shape[0] > m.shape[0]: #If female is bigger that male, reduce the length of female

                    f = f.sample(m.shape[0], random_state = 3)

                    print(f"f = {f.shape}")



                desease = pd.concat(objs=[m, f]) # Concat each gender

                print(f"desease = {desease.shape}")

                print(f" The desease are proportional: {(f.shape[0] + m.shape[0] == desease.shape[0]) & ((f.shape[1] + m.shape[1])/2 == desease.shape[1])}")



                proportion_data = pd.concat(objs=[proportion_data, desease]) #Store the desease in the new DataFrame

                print(f"Actual shape of the new DF: {proportion_data.shape}")

            else:

                print(f"Not enough m = {m.shape}, f = {f.shape} ")

                

    else:

        print(f"Not enough images to split: desease={desease.shape}")
print(f"The new shape: {proportion_data.shape}")

proportion_data.sample(4, random_state=2)
"""

Diference between original labels and new quantity of labels

"""

data_labels = [x for x in np.unique(data['Finding Labels'])]

proportion_labels = [x for x in np.unique(proportion_data['Finding Labels'])]

print(f"Original multilabels: {len(data_labels)}, Actual multilabels: {len(proportion_labels)}")
"""

Replace data with proportion_data

"""

data = proportion_data
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

(final_no_finding_data.shape[0] + data.shape[0], 

 (final_no_finding_data.shape[1] + data.shape[1])/2) == data.shape
print(data.shape)

# Show a sample of the dataset

data.sample(3, random_state=3)
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

data.head()
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

data.head()
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





new_size = 80000 #Of the dataframe



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
"""

creating vector of diseases (put a 1.0 in the order of the decease)

"""

data['disease_vec'] = data.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
data[['Image Index', 'Finding Labels', 'disease_vec']].sample(3, random_state=3)
print(data.iloc[0:5]['disease_vec'])
print(data.shape)

data[[  'Finding Labels',

        'Atelectasis',

        'Cardiomegaly',

        'Consolidation',

        'Edema',

        'Effusion',

        'Emphysema',

        'Fibrosis',

        'Hernia',

        'Infiltration',

        'Mass',

        'Nodule',

        'Pleural_Thickening',

        'Pneumonia',

        'Pneumothorax',

        'disease_vec']].sample(5, random_state=3)
data.info()
data.columns
data.to_pickle('data-chest-x-ray-multilabel-balanced-resample-gender-proportion.plk')