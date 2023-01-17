import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import missingno as misn

import pandas_profiling as pp

from subprocess import check_output

print(check_output(['ls', '../input']).decode('utf-8'))
train = pd.read_csv('../input/train.csv')
misn.bar(train)
pp.ProfileReport(train)