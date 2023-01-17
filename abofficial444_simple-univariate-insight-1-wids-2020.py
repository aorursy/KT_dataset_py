import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Linear Algebra

import numpy as np

#Data preprocessing

import pandas as pd



#setting display options

pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 500)

pd.set_option('max_colwidth', 500)

np.set_printoptions(linewidth =400)



from matplotlib import pyplot as plt

%matplotlib inline

#Advance-style plotting

import seaborn as sns

color =sns.color_palette()

sns.set_style('darkgrid')



#Ignore annoying warning from sklearn and seaborn

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn



#other libraiaries

import os

import copy

from collections import defaultdict

from collections import Counter

from sklearn import metrics

import matplotlib.pyplot as plt

%matplotlib inline

import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

import re

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
description = pd.read_csv('/kaggle/input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv'); description
#read the data

train = pd.read_csv('/kaggle/input/widsdatathon2020/training_v2.csv')

test  = pd.read_csv('/kaggle/input/widsdatathon2020/unlabeled.csv')
train.columns
print(train.shape , test.shape)
#column 1 Unique identifier associated with a patient unit stay

print (train['encounter_id'].nunique() , test['encounter_id'].nunique())
#column 2 Unique identifier associated with a hospital

print (train['hospital_id'].nunique() , test['hospital_id'].nunique())
#column 3 Unique identifier associated with a patient

print (train['patient_id'].nunique() , test['patient_id'].nunique())
#column 4

Yes = len(train[train.hospital_death ==1])

No = len(train[train.hospital_death ==0])

Total = len(train)

print ('There are imbalanace datset with a %i/%i ratio'%((No/Total*100), (Yes/Total*100)+1))
sns.catplot(x ='hospital_death', kind ='count',palette='pastel', data = train);
#columnn 5

train['age'].describe()
#Hint: ensure all units of each columns are having relationship with respect to each other..