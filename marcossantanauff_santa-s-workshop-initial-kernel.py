%matplotlib inline

import os

import pandas as pd

import numpy as np

from fastai.tabular import *

import matplotlib.pyplot as plt

import seaborn as sns
path = Path('../input/santa-workshop-tour-2019')
path.ls()
df = pd.read_csv(path/'family_data.csv')

samplesub = pd.read_csv(path/'sample_submission.csv')
df.head()
grouped_families = df.groupby('n_people')['family_id'].count().reset_index()
grouped_families.head()
sns.barplot(x='n_people',y='family_id',data=grouped_families)