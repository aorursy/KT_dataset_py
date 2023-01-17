# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))







# Any results you write to the current directory are saved as output.
df= pd.read_csv('../input/314-nps/314nps.csv')

df2 = pd.read_csv('../input/dashid2/inMomentnps idindex.csv')

df.rename(columns = {'Helpfulness of the Staff': 'HOS', 'Speed of Service' :'SOS', 'Group by Pickup CSR Number' :'CSR'}, inplace = 'True')

df.drop(columns = {'Main Hierarchy', 'Main Hierarchy.1'}, inplace = True)








