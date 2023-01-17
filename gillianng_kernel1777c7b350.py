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
data = pd.read_csv("/kaggle/input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv")

data = data[["incident_id", "date", "n_killed", "n_injured", "latitude", "longitude", "notes"]]

year = pd.to_datetime(data.date).dt.year

mask = (year >= 2016) & (year <= 2019)

data = data.loc[mask]
os.chdir(r'/kaggle/working')

data.to_json(r'gun_data.json', orient='records')
from IPython.display import FileLink

FileLink(r'gun_data.json')