!pip install pyfpgrowth
import cv2

import time

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm_notebook

import pyfpgrowth as fpg

import matplotlib.pyplot as plt
path = '../input/understanding_cloud_organization/'
train = pd.read_csv('{}//train.csv'.format(path))
tr_image_path = '{}//train_images//'.format(path)
print('Training Data shape {}'.format(train.shape))
train.head()
train[['Image_ID','Image_Label']] = train.Image_Label.str.split('_', expand=True) 
train.head()
train.to_csv('train_processsed.csv')
# creating another dataframe with redudant label entries removed

labelcount = train[['Image_ID', 'Image_Label', 'EncodedPixels']].groupby('Image_ID').apply(lambda x: x.dropna()['Image_Label'].values).reset_index()

labelcount = labelcount.rename(columns = {0: 'labels'})

labelcount['label_counts'] = labelcount['labels'].apply(lambda x: len(x))            
labelcount.head()
def get_hist(df, col):

    ax = df[col].value_counts().plot(kind = 'bar', figsize=(10,7),

                                        fontsize=10);

    ax.set_alpha(0.8)



    # create a list to collect the plt.patches data

    totals = []



    # find the values and append to list

    for i in ax.patches:

        totals.append(i.get_width())



    # set individual bar lables using above list

    total = sum(totals)



    # set individual bar lables using above list

    for i in ax.patches:

        # get_width pulls left or right; get_y pushes up or down

        ax.text(i.get_x()+.1, i.get_height()+.5, str(i.get_height()), fontsize=15,

    color='black')

        

    return ax
ax = get_hist(labelcount, 'label_counts')

ax.set_title("Histogram of Label Counts per Image", fontsize=18)

ax.set_xlabel("Number of occurrences", fontsize=18)

plt.show()
ax = get_hist(train.dropna(), 'Image_Label')

ax.set_title("Histogram of Labels (with valid masks)", fontsize=18)

ax.set_xlabel("Label", fontsize=18);
patterns = fpg.find_frequent_patterns(labelcount['labels'], 2)

patternsdf = pd.DataFrame({'Label Association': list(patterns.keys()), 'Occurrences': list(patterns.values())})
f = plt.figure(figsize = (15,10))

ax = patternsdf.plot(x = 'Label Association', y = 'Occurrences', kind = 'bar')

for i in ax.patches:

    ax.text(i.get_x()-0.2, i.get_height()+.5, str(i.get_height()), fontsize=10, color='black')

plt.show()
rules = fpg.generate_association_rules(patterns, 0.3)

rulesdf = pd.DataFrame({'Association Rules': list(rules.keys()), 

                        'Labels': [0]*len(list(rules.keys())), 'Probabilities': list(rules.values())})

rulesdf.loc[:, 'Labels'] = rulesdf['Probabilities'].apply(lambda x: x[0][0])

rulesdf.loc[:, 'Probabilities'] = rulesdf['Probabilities'].apply(lambda x: x[1])

rulesdf = rulesdf.sort_values('Probabilities', ascending = False)
rulesdf
# run length encoding function

def rle_decode(mask,shape=(1400,2100)):

    

    s=mask.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts-=1

    end=starts+lengths

    img=np.zeros(shape[0]*shape[1],dtype=np.uint8)

    for l,m in zip(starts,end):

        img[l:m]=1

    return img.reshape(shape[0],shape[1],order='F')
train.loc[:, 'MaskArea'] = train['EncodedPixels'].apply(lambda x: np.sum(rle_decode(str(x))) if not pd.isna(x) else 0)
# distribution plots for mask areas for different labels

f, ax = plt.subplots(1, 1, figsize = (10, 7))

sns.distplot(train[(train['Image_Label'] == 'Fish') & 

                   (train['MaskArea'] > 0)]['MaskArea'], kde=True, hist=False, ax = ax, color = 'red')

sns.distplot(train[(train['Image_Label'] == 'Flower') & 

                   (train['MaskArea'] > 0)]['MaskArea'], kde=True, hist=False, ax = ax, color = 'blue')

sns.distplot(train[(train['Image_Label'] == 'Gravel') & 

                   (train['MaskArea'] > 0)]['MaskArea'], kde=True, hist=False, ax = ax, color = 'green')

sns.distplot(train[(train['Image_Label'] == 'Sugar') & 

                   (train['MaskArea'] > 0)]['MaskArea'], kde=True, hist=False, ax = ax, color = 'black')

ax.legend(labels=['Fish', 'Flower', 'Gravel', 'Sugar'])

plt.show()