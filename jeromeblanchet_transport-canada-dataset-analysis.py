import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
print('Loading datasets...')

file_path = '../input/canada-aircraft-accidents/'

CADORS_Aircraft_Event_Information = pd.read_csv(file_path + 'CADORS_Aircraft_Event_Information.csv', sep=',')

CADORS_Occurrence_Category = pd.read_csv(file_path + 'CADORS_Occurrence_Category.csv', sep=',')

CADORS_Occurrence_Information = pd.read_csv(file_path + 'CADORS_Occurrence_Information.csv', sep=',')

CADORS_Occurrence_Event_Information = pd.read_csv(file_path + 'CADORS_Occurrence_Event_Information.csv', sep=',')

CADORS_Aircraft_Information = pd.read_csv(file_path + 'CADORS_Aircraft_Information.csv', sep=',')

print('Datasets loaded')
DATA1 = CADORS_Aircraft_Event_Information

DATA2 = CADORS_Occurrence_Category 

DATA3 = CADORS_Occurrence_Information 

DATA4 = CADORS_Occurrence_Event_Information 

DATA5 = CADORS_Aircraft_Information 
DATA1
DATA2
DATA3
DATA4
DATA5
DATA5.columns
DATA3.columns