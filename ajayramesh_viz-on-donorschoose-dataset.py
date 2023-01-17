# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json # for json processing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
resources = pd.read_csv('../input/Resources.csv',error_bad_lines = False, warn_bad_lines = False, index_col="Project ID")

schools = pd.read_csv('../input/Schools.csv',error_bad_lines = False, warn_bad_lines = False, index_col="School ID")

donors = pd.read_csv('../input/Donors.csv',error_bad_lines = False, warn_bad_lines = False, index_col="Donor ID", low_memory=False)

teachers = pd.read_csv('../input/Teachers.csv',error_bad_lines = False, warn_bad_lines = False, index_col="Teacher ID",  low_memory=False)

projects = pd.read_csv('../input/Projects.csv',error_bad_lines = False, warn_bad_lines = False,index_col="Project ID",low_memory=False)
r = list(resources.columns.values)

s = list(schools.columns.values)

d = list(donors.columns.values)

t = list(teachers.columns.values)

p = list(projects.columns.values)
col_names = ["Resources", "Schools", "Donors", "Teachers", "Projects"]

col_indices = [r, s, d, t, p]

col_name_indices = dict(zip(col_names,col_indices))
for key, value in col_name_indices.items():

    print(key)

    for each in value:

        print("\t", each)