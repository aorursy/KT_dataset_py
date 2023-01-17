# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_data = pd.read_csv('/kaggle/input/chicago-crime-for-da-train-question/Crimes_-_2001_to_present.csv', dtype={'ID': int, 'Case Number': object,'Date': object, 'Block': object, 'IUCR': object,'Primary Type': object,'Description': object,'Location, Description':object, 'Arrest': bool,'Domestic': bool,'Beat': int,'District': float,'Ward': float,'Community Area': float,'FBI Code': object,'X Coordinate': float,'Y Coordinate': float,'Year': int,'Updated On': object,'Latitude': float, 'Longitude': float,'Location': object})
raw_data.head()
raw_data.info()
raw_data = raw_data.drop(['ID', 'Case Number', 'Date', 'IUCR', 'X Coordinate', 'Y Coordinate', 'Updated On','Latitude', 'Longitude', 'Location'], axis = 1)
raw_data.head()
raw_data.describe()
raw_data.describe(include=[np.object])
raw_data.describe(include=[np.bool])
raw_data['Arrest'] = raw_data['Arrest'].map( {True: 1, False: 0} ).astype(int)
raw_data['Domestic'] = raw_data['Domestic'].map( {True: 1, False: 0} ).astype(int)
raw_data.head()
raw_data['Primary Type'].unique()
raw_data['Location Description'].unique()