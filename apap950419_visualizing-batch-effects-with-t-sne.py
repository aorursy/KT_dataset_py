# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from PIL import Image as image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import tqdm



!pip install opentsne

from openTSNE.sklearn import TSNE

# Any results you write to the current directory are saved as output.
train_controls = pd.read_csv('../input/train_controls.csv')

test_controls = pd.read_csv('../input/test_controls.csv')

train_controls['cell_line'] = [v[0] for v in train_controls.id_code.str.split('-')]

test_controls['cell_line'] = [v[0] for v in test_controls.id_code.str.split('-')]



train_controls.shape, test_controls.shape
train_controls.head(3)
test_controls.head(3)
positive_sirnas = train_controls[train_controls.well_type == 'positive_control'].sirna.unique()

negative_sirna = train_controls[train_controls.well_type == 'negative_control'].sirna.unique()[0]
negative_sirna
positive_sirnas
len(positive_sirnas)
def get_image_file(experiment, plate, well, channel, site=1, train=True):

    train_path = f'../input/train/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png'

    test_path = f'../input/test/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png'

    if os.path.exists(train_path):

        return train_path

    else:

        return test_path

    

def get_image_nparray(experiment, plate, well, channel, site=1, train=True):

    img = np.array(image.open(get_image_file(experiment, plate, well, channel, site, train)))

    return img.reshape(-1)
data = []

for i, row in tqdm.tqdm(train_controls.iterrows(), total=len(train_controls)):

    v = []

    for channel in range(1, 7):

        # Take means of pixel intensities.

        v.append(get_image_nparray(experiment=row.experiment, plate=row.plate, well=row.well, channel=channel).mean())

    data.append(v)

data = np.array(data)
data.shape
embedding = TSNE().fit_transform(data)
train_controls.head(3)
train_controls.cell_line.unique()
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

print('There are %d experiments.' % len(train_controls[train_controls.cell_line.values == 'RPE'].experiment.unique()))

for exp in train_controls.experiment.unique():

    mask = (train_controls.experiment.values == exp) & (train_controls.cell_line.values == 'RPE')

    if sum(mask) != 0:

        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=exp, s=6)



ax.legend();
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

print('There are %d experiments.' % len(train_controls[train_controls.cell_line.values == 'HEPG2'].experiment.unique()))



for exp in train_controls.experiment.unique():

    mask = (train_controls.experiment.values == exp) & (train_controls.cell_line.values == 'HEPG2')

    if sum(mask) != 0:

        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=exp, s=6)



ax.legend();
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)



mask = (train_controls.cell_line.values == 'HUVEC')

print('There are %d experiments.' % len(train_controls[mask].experiment.unique()))

ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[float(v[1]) / 16 for v in train_controls[mask].experiment.str.split('-')], cmap=plt.cm.Blues, s=4);
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

print('There are %d experiments.' % len(train_controls[train_controls.cell_line.values == 'U2OS'].experiment.unique()))



for exp in train_controls.experiment.unique():

    mask = (train_controls.experiment.values == exp) & (train_controls.cell_line.values == 'U2OS')

    if sum(mask) != 0:

        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=exp, s=4)



ax.legend();
print('There are %d experiments.' % len(train_controls.experiment.unique()))



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

for cl in train_controls.cell_line.unique():

    mask = (train_controls.cell_line.values == cl)

    ax.scatter(embedding[mask, 0], embedding[mask, 1], label=cl, s=4)



ax.legend()

print('There are %d experiments.' % len(train_controls.experiment.unique()))



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

for si in train_controls.sirna.unique()[:10]:

    mask = (train_controls.sirna.values == si)

    ax.scatter(embedding[mask, 0], embedding[mask, 1], label=si, s=4)



ax.legend();
print('There are %d experiments.' % len(train_controls.experiment.unique()))



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)



mask = (train_controls.sirna.values == 1138)

ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.11, color='grey', s=3)

ax.scatter(embedding[mask, 0], embedding[mask, 1], label=1138, s=8, color='red')



ax.legend();
test_controls['cell_line'] = [v[0] for v in test_controls.id_code.str.split('-')]

controls = pd.concat([train_controls, test_controls])
controls.shape
data = []

for i, row in tqdm.tqdm(controls.iterrows(), total=len(controls)):

    v = []

    for channel in range(1, 7):

        # Take means of pixel intensities.

        v.append(get_image_nparray(experiment=row.experiment, plate=row.plate, well=row.well, channel=channel).mean())

    data.append(v)



data = np.array(data)
embedding = TSNE().fit_transform(data)
CELL_LINE = 'RPE'



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)



unique_exps = controls[controls.cell_line.values == CELL_LINE].experiment.unique()

print('There are %d experiments.' % len(unique_exps))



for exp in unique_exps:

    mask = (controls.experiment.values == exp) & (controls.cell_line.values == CELL_LINE)

    if sum(mask) != 0:

        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=exp, s=6)



ax.legend();
CELL_LINE = 'HEPG2'



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)



unique_exps = controls[controls.cell_line.values == CELL_LINE].experiment.unique()

print('There are %d experiments.' % len(unique_exps))



for exp in unique_exps:

    mask = (controls.experiment.values == exp) & (controls.cell_line.values == CELL_LINE)

    if sum(mask) != 0:

        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=exp, s=6)



ax.legend();
CELL_LINE = 'HUVEC'



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)



unique_exps = controls[controls.cell_line.values == CELL_LINE].experiment.unique()

print('There are %d experiments.' % len(unique_exps))



for exp in unique_exps:

    mask = (controls.experiment.values == exp) & (controls.cell_line.values == CELL_LINE)

    if sum(mask) != 0:

        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=exp, s=6)



ax.legend();
CELL_LINE = 'U2OS'



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)



unique_exps = controls[controls.cell_line.values == CELL_LINE].experiment.unique()

print('There are %d experiments.' % len(unique_exps))



for exp in unique_exps:

    mask = (controls.experiment.values == exp) & (controls.cell_line.values == CELL_LINE)

    if sum(mask) != 0:

        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=exp, s=6)



ax.legend();