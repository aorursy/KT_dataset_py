%matplotlib inline
import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from __future__ import division

sns.set_context('notebook', font_scale=1.25)
astronauts_df = pd.read_csv('../input/astronauts.csv',

                            index_col=0,

                            skipinitialspace=True,

                            dtype={'Year': object,

                                   'Group': object},

                            parse_dates=['Birth Date', 'Death Date'],

                            na_values='')

astronauts_df.info()
undergrad_majors = astronauts_df['Undergraduate Major'].str.split(r';\s')

first_major =pd.Series(index=undergrad_majors.index.values,

                       data=[x[0] if type(x) != float else np.nan for x in undergrad_majors])

second_major =pd.Series(index=undergrad_majors.index.values,

                        data=[x[1] if type(x) != float and len(x) == 2 else np.nan for x in undergrad_majors])

majors = first_major.value_counts().add(second_major.value_counts(), fill_value=0).divide(335).sort_values(ascending=False)*100

majors.name = 'Undergraduate Major' 
print('Top 10 Most Common Undergrad Majors (Normalized %)')

print('='*10)

print(majors[:10])

print('='*10)

print('Proportion of Astronauts w/ Top 5 Most Common Undergrad Majors {0:.3f}%'.format(np.sum(majors[:5])))

print('Proportion of Astronauts w/ Top 6-10 Most Common Undergrad Majors {0:.3f}%'.format(np.sum(majors[5:10])))
grad_majors = astronauts_df['Graduate Major'].str.split(r';\s')

first_major = pd.Series(index=grad_majors.index.values,

                        data=[x[0] if type(x) != float else np.nan for x in grad_majors])

second_major = pd.Series(index=grad_majors.index.values,

                         data=[x[1] if type(x) != float and len(x) > 1 else np.nan for x in grad_majors])

third_major = pd.Series(index=grad_majors.index.values,

                        data=[x[2] if type(x) != float and len(x) > 2 else np.nan for x in grad_majors])

fourth_major = pd.Series(index=grad_majors.index.values,

                         data=[x[3] if type(x) != float and len(x) > 3 else np.nan for x in grad_majors])

majors = first_major.value_counts().add(second_major.value_counts(), fill_value=0).add(third_major.value_counts(), fill_value=0).add(fourth_major.value_counts(), fill_value=0).divide(298).sort_values(ascending=False)*100

majors.name = 'Graduate Major'
print('Top 10 Most Common Grad Majors (Normalized %)')

print('='*10)

print(majors[:10])

print('='*10)

print('Proportion of Astronauts w/ Top 5 Most Common Grad Majors {0:.3f}%'.format(np.sum(majors[:5])))

print('Proportion of Astronauts w/ Top 6-10 Most Common Grad Majors {0:.3f}%'.format(np.sum(majors[5:10])))
