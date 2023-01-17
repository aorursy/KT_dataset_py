#Ignore the seaborn warnings.
#Comes from a 2016 Census code from Derik Elliot.  Just testing how to python notebook works

import warnings
warnings.filterwarnings("ignore");

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/primary_results.csv')
NH = df[df.state == 'New Hampshire']


#g = sns.FacetGrid(NH[NH.party == 'Democrat'], col = 'candidate', col_wrap = 5)
#g.map(sns.barplot, 'county', 'fraction_votes');


g = sns.FacetGrid(NH[NH.party == 'Republican'], col = 'candidate', col_wrap = 4)
g.map(sns.barplot, 'county', 'votes');
