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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as plt

import json

%matplotlib inline



with open("/kaggle/input/AgrcultureDataset.csv") as datafile:

  data = pd.read_csv(datafile)

dataframe = pd.DataFrame(data)

stat_dist=dataframe.loc[:, ['State_Name', 'District_Name','Crop']]

print(stat_dist.nunique())



formatted=dataframe.groupby(['State_Name', 'District_Name','Crop']).sum()

formatted=formatted.drop('Crop_Year', axis=1)

formatted=formatted.drop('Area', axis=1)

formatted