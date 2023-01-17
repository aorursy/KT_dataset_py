!git clone https://github.com/miykael/parcellation_fragmenter.git
!echo $PWD

!mv ./parcellation_fragmenter/docs/readme.rst ./parcellation_fragmenter/README.rst
!pip install ./parcellation_fragmenter
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as ss

import statsmodels.api as sm

import nibabel as nb # for loading surfaces

from fragmenter import RegionExtractor # for extracting regional indices

from fragmenter import Fragment # main fragmentation class

from fragmenter import colormaps # for generating new colormaps





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
verts, faces = nb.freesurfer.io.read_geometry(

  './kaggle/output/kaggle/working/parcellation_fragmenter/data/hcp/L.sphere.32k_fs_LR.surf.gii')



# Extract region-specific indices

# Likewise, you can also specific an HCP-styped Gifti Label object

# with extension .label.gii.

E = RegionExtractor.Extractor(

  './kaggle/output/kaggle/working/parcellation_fragmenter//data/freesurfer/fsaverage/label/lh.aparc.annot')

parcels = E.map_regions()



# Define set of regions of interest.  These region names are dependent

# on the region IDs in the original annotation / label file.

rois=['temporalpole','inferiortemporal','middletemporal',

  'superiortemporal', 'transversetemporal','bankssts',

      'inferiorparietal','supramarginal']
[x[0] for x in os.walk(dirname)]