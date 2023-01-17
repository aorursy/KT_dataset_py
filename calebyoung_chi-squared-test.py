# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#import data
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
### NUMBER OF SECTIONS TO DIVIDE ID INTO 
num_of_id_buckets = 4
# Cut id's into equal segments
data['id'] = pd.qcut(data['id'], q=num_of_id_buckets)
id_data = data.id
print(id_data)
# map diagnosis
diagnosis_data = data['diagnosis'].map({'M':1,'B':0})

from scipy.stats import chi2_contingency

cont_table = pd.crosstab(id_data,diagnosis_data,margins = False)

print('Contingency table: left column represents the four possible ranges for Ids \n \n',cont_table,'\n \n')

stat, p, dof, expected = chi2_contingency(cont_table)

print("RESULTS OF CHI SQUARED TEST: \n stat: {} p_value: {} dof: {} expected {}".format(stat, p, dof, expected ))