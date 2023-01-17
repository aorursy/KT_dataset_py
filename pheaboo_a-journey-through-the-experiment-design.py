import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import sys

import matplotlib.pyplot as plt

%matplotlib inline
test_controls = pd.read_csv('../input/test_controls.csv')

test = pd.read_csv('../input/test.csv')

test['sirna'] = 0

test['well_type'] = 'treatment'

test['dataset'] = 'test'

test_controls['dataset'] = 'test'





train_controls = pd.read_csv('../input/train_controls.csv')

train = pd.read_csv('../input/train.csv')

train['sirna'] = 0

train['well_type'] = 'treatment'

train['dataset'] = 'train'

train_controls['dataset'] = 'train'



md = pd.concat([train, train_controls, test, test_controls]).reset_index(drop=True)
unique_experiments = md.experiment.unique()

len(unique_experiments)
names, counts = np.unique([experiment.split('-')[0] for experiment in unique_experiments], return_counts=True)

for experiment_name, experiment_count in zip(names, counts):

    print('{} experiments focused on the cell type {}'.format(experiment_count, experiment_name))
one_experiment_md = md[(md.experiment=='RPE-03')]

one_experiment_md.head()
one_experiment_md.groupby(['plate']).count()
import geopandas as gpd

from shapely.geometry import Point



def letter_to_int(letter):

    alphabet = list('abcdefghijklmnopqrstuvwxyz'.upper())

    return alphabet.index(letter) + 1



def well_to_point(well):

    letter = letter_to_int(well[0])

    number = int(well[1:])

    return Point(letter, number)

md['geometry'] = md.well.apply(lambda well: well_to_point(well))

md = gpd.GeoDataFrame(md)

md.head()
def plot_well_type_positions_for_experiment(experiment_name):

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for plate, ax in zip([1,2,3,4], axes):

        one_plate_md = md[md.experiment==experiment_name]

        if plate==1:

            legend=True

        else:

            legend=False

        one_plate_md.plot(column='well_type',legend=legend, ax=ax);

        if plate==1:

            leg = ax.get_legend()

            leg.set_bbox_to_anchor((-0.3, 0., 0.2, 0.2))

    _ = fig.suptitle('Plate 1 to 4 - Experiment {}'.format(experiment_name))

    

plot_well_type_positions_for_experiment(md.experiment.unique()[0])

plot_well_type_positions_for_experiment(md.experiment.unique()[-1])
def plot_sirna_positions_for_experiment(experiment_name):

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for plate, ax in zip([1,2,3,4], axes):

        one_plate_md = md[md.experiment==experiment_name]

        one_plate_md = one_plate_md[one_plate_md.well_type=='positive_control'] 

        one_plate_md.plot(column='sirna', ax=ax, categorical=True, cmap='tab20c');

    _ = fig.suptitle('Plate 1 to 4 - Experiment {}'.format(experiment_name))

    

plot_sirna_positions_for_experiment(md.experiment.unique()[0])

plot_sirna_positions_for_experiment(md.experiment.unique()[-1])