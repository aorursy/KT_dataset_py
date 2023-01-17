# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# loading required modules

import pandas as pd

import pprint

pp = pprint.PrettyPrinter(indent=4)

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

import math

from sklearn import svm

%matplotlib inline



#List Files

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# setting pandas env variables to display max rows and columns

pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows',1000)





# load 2015 US Bureau of Labor Statistics http://www.bls.gov/cps/cpsaat39.xlsx data

print("Loading.....")



historical = pd.read_csv("../input/historical.csv")

pass_06_13 = pd.read_csv("../input/pass_06_13.csv")

pass_12_13 = pd.read_csv("../input/pass_12_13.csv")



print("Finished.")
pass_12_13
pass_a_12 = pass_12_13.loc[pass_12_13['year'] == 2012]

pass_a_13 = pass_12_13.loc[pass_12_13['year'] == 2013]
pass_12
pass_b_06 = pass_06_13.loc[pass_06_13['year'] == 2006]

pass_b_07 = pass_06_13.loc[pass_06_13['year'] == 2007]

pass_b_08 = pass_06_13.loc[pass_06_13['year'] == 2008]

pass_b_09 = pass_06_13.loc[pass_06_13['year'] == 2009]

pass_b_10 = pass_06_13.loc[pass_06_13['year'] == 2010]

pass_b_11 = pass_06_13.loc[pass_06_13['year'] == 2011]

pass_b_12 = pass_06_13.loc[pass_06_13['year'] == 2012]

pass_b_13 = pass_06_13.loc[pass_06_13['year'] == 2013]
cols = ['total','female', 'female_passed', 'black', 'black_passed', 'black_male', 'black_male_passed', 'black_female', 'black_female_passed', 'hispanic', 'hispanic_passed', 'hispanic_female', 'hispanic_female_passed', 'hispanic_male', 'hispanic_male_passed', 'white', 'white_passed', 'white_male',	'white_female',	'asian',	'asian_passed',	'asian_male',	'asian_female',	'male',	'male_passed']

pass_b_06[cols] = pass_b_06[cols].replace('*', -1).replace('\d+\*', -2, regex=True).fillna(-1).astype(float)

pass_b_06[cols] = 100*pass_b_06[cols].divide(t['total'], axis='index')

pass_b_06[['state']+cols]
