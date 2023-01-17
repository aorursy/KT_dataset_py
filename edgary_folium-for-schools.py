# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import folium

from folium.plugins import HeatMap

private_df = pd.read_csv('/kaggle/input/us-schools-dataset/Private_Schools.csv')

public_df = pd.read_csv('/kaggle/input/us-schools-dataset/Public_Schools.csv')
public_df.info()
private_df.info()
#PRIVATE SCHOOL MAP

private_df = private_df[['LATITUDE','LONGITUDE']]

private_df.dropna(axis=0, subset=['LATITUDE','LONGITUDE'])

heat_data = [[row['LATITUDE'],row['LONGITUDE']] for index, row in private_df.iterrows()]



private_map = folium.Map([39.358, -98.118], zoom_start=5)

HeatMap(heat_data).add_to(private_map)



private_map
#PUBLIC SCHOOL MAP

public_df = public_df[['LATITUDE','LONGITUDE']]

public_df.dropna(axis=0, subset=['LATITUDE','LONGITUDE'])

heat_data2 = [[row['LATITUDE'],row['LONGITUDE']] for index, row in public_df.iterrows()]



public_map = folium.Map([39.358, -98.118], zoom_start=5)

HeatMap(heat_data2).add_to(public_map)



public_map