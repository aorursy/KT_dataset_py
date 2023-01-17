import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats



from scipy.stats import ttest_ind

from subprocess import check_output



cereal_df = pd.read_csv('../input/cereal.csv')





values_dict = {'N': 'Nabisco', 'Q': 'Quaker Oats', 'K': 'Kelloggs', 'R': 'Raslston Purina', 'G': 'General Mills' , 'P' :'Post' , 'A':'American Home Foods Products'}

cereal_df['mfr_name'] = cereal_df['mfr'].map(values_dict)
from scipy.stats import chisquare

scipy.stats.chisquare(cereal_df['mfr_name'].value_counts())
scipy.stats.chisquare(cereal_df['type'].value_counts())
contingency_table = pd.crosstab(cereal_df["mfr_name"],

                              cereal_df["type"])



scipy.stats.chi2_contingency(contingency_table)
sns.countplot(y = 'mfr_name' , hue = 'type' , data=cereal_df , palette ='cool')
