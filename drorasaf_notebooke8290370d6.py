# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



print('done')

# Any results you write to the current directory are saved as output.



path = "../input"



aff_asylum = pd.read_csv(os.path.join(path, "affirmative_asylum.csv"))

asylum_statistics = pd.read_csv(os.path.join(path, "asylum_statistics.csv"))

def_asylum = pd.read_csv(os.path.join(path, "defensive_asylum.csv"))

refugee_statistics = pd.read_csv(os.path.join(path, "refugee_statistics.csv"))

refugee_status = pd.read_csv(os.path.join(path, "refugee_status.csv"))



continent_map = {

    'africa': ['Algeria', 'Angola'],

    'asia': ['Afghanistan', 'Azerbaijan', ],

    'north america': [],

    'south amertica': [],

    'europe': [ 'Albania', 'Armenia', ]

    

}

refugee_statistics