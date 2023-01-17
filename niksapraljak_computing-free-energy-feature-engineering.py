# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import general toolkits

import json

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np 

import random

from tqdm import tqdm

import os 


!pip install --upgrade pip

!pip install rna-tools

!conda install -y -c bioconda forgi 

!conda install -y -c bioconda viennarna

!pip install nglview
# important sequence toolkits 

import forgi.visual.mplotlib as fvm

import forgi

import nglview

from rna_tools import Seq

from rna_tools import SecondaryStructure

import rna_tools.Seq as Seq


path = '../input/stanford-covid-vaccine/'



train = pd.read_json(path + 'train.json', lines = True)

test = pd.read_json(path + 'test.json', lines = True)

sample_df = pd.read_csv(path + 'sample_submission.csv')

free_energy = np.zeros((1,len(train.sequence)))

for sample in range(len(train.sequence)):

    seq = Seq.RNASequence(train.sequence[sample])

    free_energy[0, sample] = seq.predict_ss("RNAfold", constraints = train.structure[sample])[-7:-1]

    

free_energy_ts = np.zeros((1,len(test.sequence)))

for sample in range(len(test.sequence)):

    seq = Seq.RNASequence(test.sequence[sample])

    free_energy_ts[0, sample] = seq.predict_ss("RNAfold", constraints = test.structure[sample])[-7:-1]

    

    
plt.figure(figsize = (15,5))



plt.subplot(1,2,1)

x1 = np.linspace(0,2400, 2400)

plt.scatter(x1, free_energy, alpha = 0.3, color = 'b', label = 'train samples')

plt.xlabel('RNA ID')

plt.ylabel('Free Energy')







plt.subplot(1,2,2)

x2 = np.linspace(0,len(test.sequence), len(test.sequence))

plt.scatter(x2, free_energy_ts, alpha = 0.3, color = 'r', label = 'test sample')

plt.xlabel('RNA ID')

plt.ylabel('Free Energy')

plt.show()
free_energy_df = pd.DataFrame(free_energy[0,:], columns = ['Fenergy'])

free_energy_ts_df = pd.DataFrame(free_energy_ts[0,:], columns = ['Fenergy'])

free_energy_df.head(), free_energy_ts_df.head()





train_df = pd.concat([train,free_energy_df], axis = 1)

test_df = pd.concat([test,free_energy_ts_df], axis = 1)



display(train_df.head())

display(test_df.head())