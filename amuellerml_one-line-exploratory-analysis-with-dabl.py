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
from glob import glob

dfs = []

for file in sorted(glob('/kaggle/input/cicids2017/MachineLearningCSV/MachineLearningCVE/*.csv')):

    dfs.append(pd.read_csv(file))

# data = pd.concat(dfs)

data = pd.read_csv('/kaggle/input/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv')
!pip install dabl==0.1.6
data.columns
from dabl import plot

plot(data, target_col=' Label', type_hints={' Label': 'categorical'})