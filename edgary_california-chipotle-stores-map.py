# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import folium



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/chipotle-locations/chipotle_stores.csv')

cal_df = df[df['state'] == 'California']

cal_df.head()
cal_map = folium.Map([36.778, -119.417], zoom_start=6)

for i in range(len(cal_df)):

    folium.Marker([cal_df.iloc[i]['latitude'], cal_df.iloc[i]['longitude']], popup=cal_df.iloc[i]['address']).add_to(cal_map)

cal_map