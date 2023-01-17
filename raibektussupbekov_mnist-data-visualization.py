# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# First of all, let's read the data into pandas.DataFrame
train_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
import matplotlib.pyplot as plt

# A good explanation how to line up the x ticks with histogram:
# https://stackoverflow.com/questions/27083051/matplotlib-xticks-not-lining-up-with-histogram

counts = np.bincount(train_dataset['label'])

plt.style.use('seaborn-dark-palette')

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(range(10), counts, width=0.8, align='center')
ax.set(xticks=range(10), xlim=[-1, 10], title='Training data distribution')

plt.show()
def display_samples(trainig_dataset: pd.DataFrame,
                    digits=[0,1,2,3,4,5,6,7,8,9],
                    number_in_row=5,
                    figsize=(10,25),
                    **imshow_kwargs):
    """
    Ramdomly picks digits from dataset and displays them.
    
    Keyword arguments:
    training_dataset -- consists of 'label' column and 784 columns from pixel0 to pixel783 
    digits -- digits to display, order matters
    number_in_row -- how many samples of each digit to display in a row
    figsize -- figure size, tuple (width, height) in inches
    imshow_kwargs -- keyword arguments of matplotlib.axes.Axes.imshow() except X 
    (cmap, norm, aspect, interpolation, etc) according to the definition
    https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow
    """
    # randomly picks records
    picked_records = train_dataset[train_dataset['label'].isin(digits)].sample(frac=1).groupby('label').head(number_in_row)
    
    fig, axes = plt.subplots(len(digits), number_in_row, figsize=figsize)
    
    for i in range(len(digits)):
        i_digit_records = picked_records[picked_records['label']==digits[i]]
        
        # converts rows of pixels into 28x28 matrices
        image_array = i_digit_records.iloc[:, 1:].values.reshape(number_in_row, 28, 28)
        
        for j in range(number_in_row):
            axes[i, j].imshow(
                image_array[j, :, :],
                cmap = imshow_kwargs.get('cmap', None),
                norm = imshow_kwargs.get('norm', None),
                aspect = imshow_kwargs.get('aspect', None),
                interpolation = imshow_kwargs.get('interpolation', None),
                alpha = imshow_kwargs.get('alpha', None),
                vmin = imshow_kwargs.get('vmin', None),
                vmax = imshow_kwargs.get('vmax', None),
                origin = imshow_kwargs.get('origin', None),
                extent = imshow_kwargs.get('extent', None),
                filternorm = imshow_kwargs.get('filternorm', 1),
                filterrad = imshow_kwargs.get('filterrad', 4.0),
                resample = imshow_kwargs.get('resample', None),
                url = imshow_kwargs.get('url', None)
            )
# In this call the function is set to display randomly picked samples:
# of these digits: 0,3,5,7,9
# 4 times in a row for each digit 
# in figure with 10 inch width and 12 inch height 
# as grayscale images
display_samples(train_dataset, [0,3,5,7,9], 4, figsize=(10,12), cmap='gray', vmin=0, vmax=255)