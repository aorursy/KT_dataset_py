# Importing required things
import os
import csv
from tqdm import tqdm
from collections import OrderedDict
from collections import Counter
import regex
import itertools
import random

from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# file locations
input_filepath = '../input'
train_csv_filepath = os.path.join(input_filepath, 'train.csv')
# labels list
# order should be the same as number in dataset
labels = ["Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center", "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus", 
          "Peroxisomes", "Endosomes", "Lysosomes", "Intermediate filaments", "Actin filaments", "Focal adhesion sites", "Microtubules", "Microtubule ends", "Cytokinetic bridge", 
          "Mitotic spindle", "Microtubule organizing center", "Centrosome", "Lipid droplets", "Plasma membrane", "Cell junctions", "Mitochondria", "Aggresome", "Cytosol", "Cytoplasmic bodies", "Rods & rings"]
def parse_indice(indice):
    """
    Converts various types of indice lists into normal Python list of integers
    """
    # case of space separated string, as in dataset
    if isinstance(indice, str):
        indice = indice.split(' ')
        indice = [int(index) for index in indice]
        
    # scalar converted to single-element list
    if not isinstance(indice, list):
        indice = [indice]
        
    return indice
def get_labels(indice):
    """
    Returns the labels of given indice. Indice may be give as space separated string (as in dataset), as python list or as single value
    """

    ans = [labels[index] for index in parse_indice(indice)]
    return ans  

# def try_get_labels():
#     arguments = [0, [0], '1 5']
#     return [get_labels(arg) for arg in arguments]
# try_get_labels()

def get_hots(indice):
    """
    Returns 1-hot representation of a given indice. 
    Since it can be multiple indice, it is actually many-hot
    """
    
    # range(len(labels)): sequence of integers from zero to number of labels-1
    # int(index in parse_indice(indice)): 1-hot computation
    ans = np.asarray([int(index in parse_indice(indice)) for index in range(len(labels))])
    return ans
    
# def try_get_hots():
#     arguments = [0, [0], '1 5']
#     return [get_hots(arg) for arg in arguments]

# np.asarray(try_get_hots())
def read_train_set():
    """
    Reads entire trainset into the list of dicts
    Images are not readen
    """
    ans = []
    
    # train set is guided by CSV file
    with open(train_csv_filepath) as fp:
        reader = csv.DictReader(fp, delimiter=',')
        # reading all rows and appending extra keys
        for row in reader:
            row['Hots'] = get_hots(row['Target'])
            row['Labels'] = get_labels(row['Target'])
            row['Train'] = True
            ans.append(row)
    return ans

# def try_read_train_set():
    
#     train_set = read_train_set();
#     return train_set[0];

# try_read_train_set()
def parse_filename(filename: str):
    """
    Extracts sample id and "color" from filename
    """
    filename = os.path.splitext(filename)[0]
    Id, Color = filename.split('_')
    return {'Id': Id, 'Color': Color}

# def try_parse_filename():
    
#     args = ['00631ec8-bad9-11e8-b2b9-ac1f6b6435d0_red.png']
#     return [parse_filename(filename) for filename in args]

# try_parse_filename()
def read_test_set():
    """
    Reads entire test set into list of dicts
    Keys are the same as for dataset, except labels are not given
    """
    
    # test set in guided by present files 
    filenames = os.listdir(os.path.join(input_filepath, 'test'))
    # unique ids
    Ids = set([parse_filename(filename)['Id'] for filename in filenames])
    # forming the same dicts as in train set
    return [OrderedDict([('Id', id), ('Train', False)]) for id in Ids]

# def try_read_test_set():
#     return read_test_set()[0]

# try_read_test_set()   
    
# dataset, including both train and test parts, one after another
dataset0 = read_train_set() + read_test_set()
# shuffling
random.shuffle(dataset0)

# dict to address samples by id
by_id_index = {Sample['Id']: i for (i,Sample) in enumerate(dataset0)}
def dataset0_get_sample(indice_or_ids):
    """
    Returns samples by given list of (ordinal) indice or string Ids
    """
    if isinstance(indice_or_ids, list):
        return [dataset0_get_sample(index) for index in indice_or_ids]

    if isinstance(indice_or_ids, str):
        indice_or_ids = by_id_index[indice_or_ids]

    return dataset0[indice_or_ids]
def dataset0_filter(Train: bool = True, Ids = None, Labels = None, AllLabels = False, LabelCount = None, Folds = None, FoldsCount: int = None):
    """
    Creates filtering generator, which returns all samples in sequence, matching conditions

    All specified conditions should be satisfied (logical AND)

    :param Train: include train set (`True`, default), test set (`False`) or both (`None`)
    :param Ids: include specified Id or Ids, can be regex; 
    default is `None`, which means include all
    :param Labels: include specified labels; `None` means any labels (Default)
    :param AllLabels: include if all label are present (True) or 
    any label is present (False, default)
    :param LabelCount: number of labels in sample
    :param Folds: numeric indice of folds to include; 
    dataset is slit into folds by hash code
    :param FoldsCount: total number of folds to split dataset into
    """

    for sample in dataset0:

        # checking train or test set parameter
        if Train is not None and sample['Train'] != Train:
            continue

        # checking ids regexes
        if Ids is not None:
            if not isinstance(Ids, list):
                Ids = [Ids]

            if not any(regex.match(sample['Id']) for regex in Ids):
                continue

        # checking labels filter
        if Labels is not None:

            if 'Labels' not in sample:
                continue

            if not isinstance(Labels, list):
                Labels = [Labels]
    
            if AllLabels:
                if not all(label in Labels for label in sample['Labels']):
                    continue
            else:
                if not any(label in Labels for label in sample['Labels']):
                    continue
                
                
        if LabelCount is not None:
            
            
            
            if 'Labels' not in sample:
                actual_count = 0
            else:
                actual_count = len(sample['Labels'])
                
            if actual_count != LabelCount:
                continue

        # checking folds parameters
        # there are two of them: folds list and folds count, working together
        # dataset is splitten into number of folds, specified by folds count
        # and then only folds specified by list returned
        if Folds is not None:

            if not isinstance(Folds, list):
                Folds = [Folds]

            # fold is computed from hash
            # by definition, hash should be random
            h = hash(sample['Id'])
            Fold = h % FoldsCount
            
            if Fold not in Folds:
                continue

        yield sample       


#dataset0_get_sample(0)
#dataset0_get_sample('00631ec8-bad9-11e8-b2b9-ac1f6b6435d0')
#dataset0_filter(Train = False).__next__()
#list(itertools.islice(dataset0_filter(Train = True),1))
#list(itertools.islice(dataset0_filter(Labels = "Focal adhesion sites"),3))
#list(itertools.islice(dataset0_filter(Folds = 0, FoldsCount = 3),1))
#list(itertools.islice(dataset0_filter(Folds = 1, FoldsCount = 3),1))
#list(itertools.islice(dataset0_filter(Folds = 2, FoldsCount = 3),1))
colors = ['red', 'green', 'blue', 'yellow']
def get_filepath(sample, color):
    """
    Computes path to image file, specified by given sample dict and color
    
    Image can loose data, so use it for visualization only
    """
    filename = '%s_%s.png' % (sample['Id'], color)
    if sample['Train']:
        return os.path.join(input_filepath, 'train', filename)
    else:
        return os.path.join(input_filepath, 'test', filename)
def get_PIL_image(sample, color):
    """
    Reads sample as PIL image
    
    Image retained pale to conserve data
    """
    filepath = get_filepath(sample, color)
    return Image.open(filepath).convert('RGB')
def get_PIL_image_colored(sample, color):
    """
    Reads sample as PIL images and colors it according to color suffix
    """
    if 'red' == color:
        matrix = (1, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 0, 0)
    elif 'green' == color:
        matrix = (0, 0, 0, 0,
              1, 0, 0, 0,
              0, 0, 0, 0)
    elif 'blue' == color:
        matrix = (0, 0, 0, 0,
              0, 0, 0, 0,
              1, 0, 0, 0)
    elif 'yellow' == color:
        matrix = (1, 0, 0, 0,
              1, 0, 0, 0,
              0, 0, 0, 0)
    return get_PIL_image(sample, color).convert('RGB', matrix)
#get_PIL_image_colored(dataset0_get_sample('0ba299c4-bbbc-11e8-b2ba-ac1f6b6435d0'), 'yellow')
def get_numpy_images(sample):
    """
    Reads all sample images as numpy 4-channel array 
    Pixel values normalized to one
    """
    images = [np.array(get_PIL_image_colored(sample, color)) for color in colors]
    images_np = np.stack(images)
    images_np = images_np / 255
    return images_np
def get_PIL_image_mixed(sample):
    """
    Reads all sample images and mixes them into single multicolor image
    
    Image can loose data, so use it for visualization only
    """
    images_np = get_numpy_images(sample) * 255
    #images_np = np.sum(images_np, axis=0) / len(images)
    images_np = np.sum(images_np, axis=0)
    images_np = images_np.astype( np.uint8 )
    ans = Image.fromarray(images_np)
    return ans
#get_PIL_image_mixed(dataset0_get_sample('0ba299c4-bbbc-11e8-b2ba-ac1f6b6435d0'))
def create_subplots(rows, cols, scale_factor=5):
    """
    Creates subplot with given number of rows and columns and a given scale
    :param rows:
    :param cols: number of columns
    :param scale_factor: to set figure size on screen or browser
    """
    fig, axes = plt.subplots(rows, cols,figsize=(cols*scale_factor,rows*scale_factor))
    return fig, axes
def plot_func_image(pil_image_func):
    plt.imshow(pil_image_func())
    plt.axis('off')
def plot_sample_title(sample):
    if 'Labels' in sample:
        title = ', '.join(sample['Labels'])
    else:
        title = 'No Labels (test set)'
    plt.title(title)
# # testing
# fig, axes = create_subplots(3,3)
# for ax, sample in zip(itertools.chain(*axes), dataset0):
#     plt.sca(ax)
#     plot_func_image(lambda: get_PIL_image(sample, 'green'))
#     plot_sample_title(sample)
# # testing
# fig, axes = create_subplots(3,3)
# for ax, sample, color in zip(itertools.chain(*axes), dataset0, itertools.cycle(colors)):
#     plt.sca(ax)
#     plot_func_image(lambda: get_PIL_image_colored(sample, color))
#     plot_sample_title(sample)
# Drawing some random images from entire dataset
fig, axes = create_subplots(3,3)
for ax, sample in zip(itertools.chain(*axes), dataset0):
    plt.sca(ax)
    plot_func_image(lambda: get_PIL_image_mixed(sample))
    plot_sample_title(sample)
# Drawing several random images per each label from train dataset 
ncols = 5
fig, axes = create_subplots(len(labels), ncols)
for row, label in enumerate(labels):
    for col, sample in zip(range(ncols), dataset0_filter(Labels=label, LabelCount=1)):
        plt.sca(axes[row,col])
        plot_func_image(lambda: get_PIL_image_mixed(sample))
        plot_sample_title(sample)
# Drawing several random images per each label from train dataset 
# removed condition to have only one label and now displaying in grayscale of green part, which will be used in recognition
ncols = 3
color = 'green'
fig, axes = create_subplots(len(labels), ncols)
for row, label in enumerate(labels):
    for col, sample in zip(range(ncols), dataset0_filter(Labels=label)):
        plt.sca(axes[row,col])
        plot_func_image(lambda: get_PIL_image(sample, color))
        plot_sample_title(sample)
# example: counting samples in train subset
len(list(dataset0_filter(Train = True)))
# numbers of samples in train and test sets
# wrapping into dataframe
pd.DataFrame(
    [(
        'Train' if train else 'Test', 
        len(list(dataset0_filter(Train = train)))
    ) for train in [True, False]],
    columns=['Subset','Count']
)
# numbers of samples with given number of labels
# we are specifying filter `Train=None` to include test set in counting
# test samples contain no labels
pd.DataFrame(
    [(
        n, 
        len(list(dataset0_filter(Train=None, LabelCount=n)))
    ) for n in range(len(labels)+1)],
    columns=['# of Labels','Count']
)
fig, axes = create_subplots(2,1)
for ax, sample in zip(axes, dataset0_filter(LabelCount=5)):
    plt.sca(ax)
    plot_func_image(lambda: get_PIL_image_mixed(sample))
    plot_sample_title(sample)
# counting labels
labels = []
for sample in dataset0_filter():
    for label in sample['Labels']:
        labels.append(label)
counts = Counter(labels)
df = pd.DataFrame.from_dict(counts)
df.plot(kind='bar')
