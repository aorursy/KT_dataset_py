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
data = pd.read_csv("/kaggle/input/ufcdata/data.csv")

data.head()
data.shape
data[data.location == 'Los Angeles, California, USA']
(data['R_Reach_cms'].mean() + data['B_Reach_cms'].mean())/2
data[['R_Reach_cms', 'B_Reach_cms']].min()
data.groupby(['R_Stance','B_Stance'])[['R_Reach_cms', 'B_Reach_cms']].mean()
cities = data.replace('None', np.NaN).groupby('location').size().sort_values(ascending=False).iloc[:5]

cities.plot.pie()