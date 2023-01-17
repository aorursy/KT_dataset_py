!pip install pandas-profiling==2.2.0
import pandas_profiling as pp

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import os



warnings.filterwarnings('ignore')

%matplotlib inline

mteams = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')

wteams = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WTeams.csv')

mseasons = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MSeasons.csv')

wseasons = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WSeasons.csv')
pp.ProfileReport(mteams)
pp.ProfileReport(wteams)
pp.ProfileReport(mseasons)
pp.ProfileReport(wseasons)