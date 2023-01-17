import numpy as np

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import os

print(os.listdir("../input"))
LS09Cand = pd.read_csv('../input/india-general-election-data-2009-and-2014/LS2009Candidate.csv')

print(LS09Cand.shape)

LS09Cand.head()
LS14Cand = pd.read_csv('../input/india-general-election-data-2009-and-2014/LS2014Candidate.csv')

print(LS14Cand.shape)

LS14Cand.head()
LS14Cand = pd.read_csv('../input/india-general-election-data-2009-and-2014/LS2009Electors.csv')

print(LS14Cand.shape)

LS14Cand.head()
LS14Cand = pd.read_csv('../input/india-general-election-data-2009-and-2014/LS2014Electors.csv')

print(LS14Cand.shape)

LS14Cand.head()