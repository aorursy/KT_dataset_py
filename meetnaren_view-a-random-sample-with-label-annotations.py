# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
PATH='../input/'
train_path=PATH+'train/train.csv'
train=pd.read_csv(train_path)
breeds=pd.read_csv('../input/breed_labels.csv')
colors=pd.read_csv('../input/color_labels.csv')
states=pd.read_csv('../input/state_labels.csv')
import matplotlib.pyplot as plt
from PIL import Image
import json

def explore_data(ix):
    petID=train.at[ix, 'PetID']
    typeID=train.at[ix, 'Type']
    animal_type=['dog', 'cat'][typeID-1]
    group=['', 's'][(train.at[ix, 'Quantity']>1)*1]
    name=train.at[ix, 'Name']
    n_images=int(train.at[ix, 'PhotoAmt'])
    gender=['male', 'female', 'group of'][train.at[ix, 'Gender']-1]
    desc=train.at[ix, 'Description']
    breed1_id=train.at[ix, 'Breed1']
    breed1_text=breeds[(breeds.BreedID==breed1_id) & (breeds.Type==typeID)]['BreedName'].iloc[0]
    breed2_id=train.at[ix, 'Breed2']
    if breed2_id>0:
        breed2_text=breeds[(breeds.BreedID==breed2_id) & (breeds.Type==typeID)]['BreedName'].iloc[0]
        breed_text=f'mixed breed ({breed1_text} + {breed2_text})'
    else:
        breed_text=breed1_text

    print(f'At index {ix} is a {gender} {breed_text} {animal_type}{group} named {name}. Description:\n{desc}\nHere are some pictures and their labels:')
    
    fig=plt.figure(figsize=(15, n_images*5))

    for i in range(n_images):
        ax = fig.add_subplot(n_images, 2, (i+1)*2-1, xticks=[], yticks=[])
        with open(f'../input/train_images/{petID}-{i+1}.jpg', 'rb') as f:
            img=Image.open(f)
            ax.imshow(img)
        ax = fig.add_subplot(n_images, 2, (i+1)*2, xticks=[], yticks=[])
        with open(f'../input/train_metadata/{petID}-{i+1}.json', 'r') as f:
            annos=json.load(f)
            t=''
            for i in annos['labelAnnotations']:
                t=t+'\n'+i['description']
            #print(annos)
            ax.text(0.5, 0.5, t, fontsize=15, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
explore_data(int(np.random.randint(len(train),size=1)))
