# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# We can try to guess the physical object types be studying their properties.
# This information will be useful, in case we want to fit the light curves with some of the proposed algorithms (see 'The PLAsTiCC Astronomy Classification Demo' by michaelapers)
# Fine tuning these algorithms will potentially speed up and improve the machine learning predictions, because the training data set is not representative of test data set.
# Loading the data
train = pd.read_csv('../input/training_set.csv')
train.name = 'Training Set'
train_meta = pd.read_csv('../input/training_set_metadata.csv')
train_meta.name = 'Training Metadata Set'
train_comp = pd.merge(train_meta, train, on='object_id')
order = [53, 64, 6, 95, 52, 67, 92, 88, 62, 15, 16, 65, 42, 90]
legend_class = train_meta['target'].value_counts().to_frame().sort_values(['target'])
legend_class.index.name = 'Class'
legend_class.columns = ['Number count']
# 14 out of 15 object classes are populated.
# The distribution of objects is certainly irregular.
sb.set(rc={'figure.figsize':(15,10)})
sb.barplot(x=train_meta['target'].value_counts().index.ravel(), y=train_meta['target'].value_counts().ravel(), label=legend_class, order=order)
plt.legend(loc='upper left')
# Let us have a look at the redshift statistics.
# We want to know the minimum, median and maximum redshifts for each of the classes.

order = [6, 16, 53, 65, 92, 64, 67, 62, 52, 15, 90, 42, 95, 88] # order originates from the maximum redshift measured
sb.set_color_codes("pastel")
max_z = train_meta.groupby('target')['hostgal_specz'].max().sort_values()
sb.barplot(x=max_z.index,
           y=max_z.values,  label="Max", color='b', order=order)
med_z = train_meta.groupby('target')['hostgal_specz'].median().sort_values()
sb.barplot(x=med_z.index,
           y=med_z.values, label="Mean", color='r', order=order)
min_z = train_meta.groupby('target')['hostgal_specz'].min().sort_values()
sb.set_color_codes("muted")
sb.barplot(x=min_z.index, 
           y=min_z.values, label="Min", color='y', order=order)
plt.legend(loc='upper left')
plt.ylabel('hostgal_specz')
plt.axis([-0.5,13.5,0,1.5])

# We learn that
# 6, 16, 53, 65, 92 have to be innergalactic objects: stars or star-like objects
# 64, 67, 62, 52, 15, 90, 42, 95, 88 are extragalactic objects (Active Galactic Nuclei, Supernovae, etc.)
# furthermore, we can identify at least two subcategories:
# 95 and 88 seem to be very distant objects (scale factor of the Universe ~ 1 / (1 + z) , for z <= 1000), these objects must be bright and present at earlier times of the Universe
# the other objects have a median redshift of 0.2
# Now we go through each of the classes (a very traditional approach ... )
# Physically correct reasoning will follow...
#comment: the sb.scatterplot(hue=...) does not seem to work properly. any ideas?

from random import randint

def plt_cls(df, cls, obj=None, hue=False, ln_plt=False):
    plt.figure(figsize=(15,5))
    obj_class = df[(df['target']==cls)]
    unique_ids = obj_class['object_id'].unique()
    if obj == None:
        obj = unique_ids[randint(0,len(unique_ids)-1)]
    label = 'class: ' + str(cls) + ', obj_id: '  + str(obj) + ', z: ' + str(list(df[df['object_id']==obj]['hostgal_photoz'])[0])
    print(label)
    mjd = obj_class[obj_class['object_id']==obj]['mjd']
    flx = obj_class[obj_class['object_id']==obj]['flux']
    if ln_plt == False:
        if hue == False:
            sb.scatterplot(mjd, flx, label=label)
        else:
            sb.scatterplot(mjd, flx, hue=df['passband'])
    else:
        sb.lineplot(mjd, flx, hue=df['passband'])
    plt.legend(loc='upper left')
    plt.plot()
# random behavior
# guess: variable star
plt_cls(train_comp, 16, hue=True)
# periodicity of about 100 days
# achromatic flux change
# guess: pulsating star
plt_cls(train_comp, 53, hue=True)
# single event
# short timescale of few days
# chromatic flux change
# guess: microlensing event
plt_cls(train_comp, 6)
# single event
# random behavior
# guess: variable star/eruptive
plt_cls(train_comp, 65, hue=True)
# random behavior & huge fluctuations
# guess:  variable star
plt_cls(train_comp, 92, hue=True)
# single event
# achromatic
# long timescale
# guess: supernova 1a
plt_cls(train_comp, 95, hue=True)
# high redshift
# variability
# long timescale
# guess: AGN
plt_cls(train_comp, 88, hue=True)
# single event
# short timescale
# guess: merging event
plt_cls(train_comp, 64, hue=True)
# single event
# symmetric
# short timescale
# guess: merging event
plt_cls(train_comp, 67, hue=True)
# single event
# long timescale
# guess: supernova type 1
plt_cls(train_comp, 62, hue=True)
# single event
# asymmetric
# guess: supernova type 2
plt_cls(train_comp, 52, hue=True)
# single event
# asymmetric
# guess: supernova type 2
plt_cls(train_comp, 15, hue=True)
# single event
# guess: supernova type 1
plt_cls(train_comp, 90, hue=True)
# single event
# short timescale
# guess: supernova type 1
plt_cls(train_comp, 42, hue=True)
import pandas as pd
# One quick addition:
#test = pd.read_csv('../input/test_set.csv', low_memory=True, chunk)
test_meta = pd.read_csv('../input/test_set_metadata.csv', low_memory=True)
# From the redshift distribution of the training and test set follows, that
# zero redshift, with redshift between 0.05 and 0.3 and around 2.5 are overrepresented in the training set
# objects with redshift between 0.3 and 1.0 are underrepresented as well -> data augmentation can be useful to avoid misclassification

plt.figure(figsize=(20,5))
sb.distplot(test_meta['hostgal_photoz'])
sb.distplot(train_meta['hostgal_photoz'])
plt.axis([0.0,1.0,0.0,2.0])
plt.figure(figsize=(20,5))
sb.distplot(test_meta['hostgal_photoz'])
sb.distplot(train_meta['hostgal_photoz'])
plt.axis([1.0,3.0,0.0,0.1])
plt.plot()
